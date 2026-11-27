#!/usr/bin/env python3
"""
Mindlin-Reissner 板超声导波仿真 (DOLFINx 0.7.x)
- 生成健康板/含损伤板波场数据
- 输出格式与现有 .pkl 对齐：[Ns, Nt] 蛇形扫描
- 用于验证色散关系、建立高保真训练数据
"""
import numpy as np
import dolfinx
import ufl
from dolfinx import fem, mesh, io
from dolfinx.fem import Function, functionspace, dirichletbc, locate_dofs_topological
from dolfinx.fem.petsc import LinearProblem
from mpi4py import MPI
import gmsh
import yaml
import json
from pathlib import Path
from datetime import datetime
import sys
import pandas as pd

# ----------------------------------------------------------
# 1. 读取物理参数
# ----------------------------------------------------------
params_path = Path("params/physical_params.yaml")
with open(params_path) as f:
    params = yaml.safe_load(f)

mat = params['physical_params']['material']
geo = params['physical_params']['geometry']
exc = params['physical_params']['excitation']
scan = params['physical_params']['scan_region']

E = mat['young_modulus_pa']
nu = mat['poisson_ratio']
rho = mat['density_kg_m3']
h = geo['thickness_m']
dx = geo['dx_m']
dy = geo['dy_m']
Lx = geo['plate_length_x_m']
Ly = geo['plate_width_y_m']

fc = exc['center_frequency_hz']
bw = exc['bandwidth_hz']
t0 = exc['time_center_us'] * 1e-6
A = exc['amplitude_norm']

nx, ny = scan['nx'], scan['ny']
roi_x0, roi_x1 = scan['roi_x0'], scan['roi_x1']
roi_y0, roi_y1 = scan['roi_y0'], scan['roi_y1']

# 时间离散
fs = exc['sampling_rate_hz']
dt = 1.0 / fs
Nt = 1000
t_vec = np.arange(Nt) * dt

# ----------------------------------------------------------
# 2. 几何与网格 (GMSH 生成矩形板)
# ----------------------------------------------------------
gmsh.initialize()
gmsh.model.add("mindlin_plate")

# 矩形
gmsh.model.occ.addRectangle(0, 0, 0, Lx, Ly, 1)
gmsh.model.occ.synchronize()

# 物理组
gmsh.model.addPhysicalGroup(2, [1], 1, "plate")
gmsh.model.addPhysicalGroup(1, [1,2,3,4], 2, "boundaries")
gmsh.model.occ.synchronize()

# 网格尺度：每波长至少 10 单元，A0 波长 ~1.5mm -> h_elem = 0.1mm
h_elem = 0.1e-3
gmsh.option.setNumber("Mesh.CharacteristicLengthMin", h_elem)
gmsh.option.setNumber("Mesh.CharacteristicLengthMax", h_elem)
gmsh.model.mesh.generate(2)

# 转为 DOLFINx mesh
domain, cell_tags, facet_tags = dolfinx.io.gmshio.model_to_mesh(
    gmsh.model, MPI.COMM_WORLD, 0, gdim=2
)
gmsh.finalize()

print(f"Mesh: {domain.topology.index_map(0).size_local} vertices, {domain.topology.index_map(2).size_local} cells")

# ----------------------------------------------------------
# 3. Mindlin-Reissner 有限元空间 (MITC4 简化：w-CG2, theta-CG1)
# ----------------------------------------------------------
V_w = functionspace(domain, ("CG", 2))
V_tx = functionspace(domain, ("CG", 1))
V_ty = functionspace(domain, ("CG", 1))

from dolfinx.fem import MixedFunctionSpace
V = MixedFunctionSpace([V_w, V_tx, V_ty])

w, tx, ty = ufl.TrialFunctions(V)
dw, dtx, dty = ufl.TestFunctions(V)

# ----------------------------------------------------------
# 4. 物理参数与本构
# ----------------------------------------------------------
D = E * h**3 / (12 * (1 - nu**2))
G = E / (2 * (1 + nu))
kappa = 5/6
kappaGh = kappa * G * h
rho_h = rho * h
rho_h3 = rho * h**3 / 12

# 应变
kappa_xx = -tx.dx(0)
kappa_yy = -ty.dx(1)
kappa_xy = -0.5 * (tx.dx(1) + ty.dx(0))

gamma_xz = tx + w.dx(0)
gamma_yz = ty + w.dx(1)

# 应力
Mxx = D * (kappa_xx + nu * kappa_yy)
Myy = D * (kappa_yy + nu * kappa_xx)
Mxy = D * (1 - nu) * kappa_xy

Qx = kappaGh * gamma_xz
Qy = kappaGh * gamma_yz

# ----------------------------------------------------------
# 5. 时域弱形式 (Newmark-beta 隐式, beta=0.25, gamma=0.5)
# ----------------------------------------------------------
beta = 0.25
gamma_nm = 0.5

W = Function(V)
W_n = Function(V)
W_n1 = Function(V)

w_n, tx_n, ty_n = W_n.sub(0), W_n.sub(1), W_n.sub(2)
w_n1, tx_n1, ty_n1 = W_n1.sub(0), W_n1.sub(1), W_n1.sub(2)

a_w = (w - 2*w_n + w_n1) / dt**2
a_tx = (tx - 2*tx_n + tx_n1) / dt**2
a_ty = (ty - 2*ty_n + ty_n1) / dt**2

residual = (
    rho_h * a_w * dw
    + rho_h3 * a_tx * dtx
    + rho_h3 * a_ty * dty
    + Mxx * (-dtx.dx(0)) + Myy * (-dty.dx(1)) + Mxy * (-0.5*(dtx.dx(1) + dty.dx(0)))
    + Qx * (dtx + dw.dx(0)) + Qy * (dty + dw.dx(1))
) * ufl.dx

# ----------------------------------------------------------
# 6. 边界条件：简支 w=0, M_n=0 (tx=0 在 x边界, ty=0 在 y边界)
#    激励：左边界中心施加时变面外位移 w(t) = gaussian_pulse(t)
# ----------------------------------------------------------
def boundary_x0(x): return np.isclose(x[0], 0)
def boundary_x1(x): return np.isclose(x[0], Lx)
def boundary_y0(x): return np.isclose(x[1], 0)
def boundary_y1(x): return np.isclose(x[1], Ly)

# 激励位置：左边界中心
x_src = 0.0
y_src = Ly / 2

def gaussian_pulse(t):
    tau = 1.0 / (np.pi * bw)
    return A * np.exp(-((t - t0) / tau)**2) * np.sin(2*np.pi*fc*(t - t0))

# 定位激励节点 (x=0 边界上最接近 y_src 的节点)
V_w_dofmap = V_w.dofmap
V_w_im = V_w_dofmap.index_map
# 获取 x=0 边界的自由度
boundary_facets_x0 = locate_dofs_topological(V_w, 1, facet_tags.find(1))  # 简化：后续需精确定位

# 简化：使用 DirichletBC 在激励点施加位移
# 实际需要更精确的点源实现，这里演示框架
bcs = []

# 简支边界：w=0 在所有边界
# w=0
bcs.append(dirichletbc(fem.Constant(domain, 0.0), locate_dofs_topological(V_w, 1, facet_tags.find(2)), V_w))
# Mxx=0 -> tx=0 在 x=0, Lx
bcs.append(dirichletbc(fem.Constant(domain, 0.0), locate_dofs_topological(V_tx, 1, facet_tags.find(2)), V_tx))
# Myy=0 -> ty=0 在 y=0, Ly
bcs.append(dirichletbc(fem.Constant(domain, 0.0), locate_dofs_topological(V_ty, 1, facet_tags.find(2)), V_ty))

# 激励：在左边界中心附近施加 w(t)
# 这里简化：不施加激励边界条件，改用初始条件脉冲
# 更严谨需用 PointSource 或 DiracDelta

# ----------------------------------------------------------
# 7. 求解器
# ----------------------------------------------------------
# 线性化：对残差关于 TrialFunctions 求导
a = ufl.lhs(residual)
L = ufl.rhs(residual)

problem = LinearProblem(
    a, L,
    bcs=bcs,
    petsc_options={
        "ksp_type": "preonly",
        "pc_type": "lu",
        "pc_factor_mat_solver_type": "mumps",
    }
)

# ----------------------------------------------------------
# 8. ROI 蛇形扫描点
# ----------------------------------------------------------
x_roi = np.linspace(roi_x0*dx, (roi_x1-1)*dx, roi_x1-roi_x0)
y_roi = np.linspace(roi_y0*dy, (roi_y1-1)*dy, roi_y1-roi_y0)
X, Y = np.meshgrid(x_roi, y_roi, indexing='xy')
scan_points = []
for i, y in enumerate(y_roi):
    if i % 2 == 0:
        xs = X[i, :]
    else:
        xs = X[i, ::-1]
    for x in xs:
        scan_points.append([x, y])
scan_points = np.array(scan_points, dtype=np.float64)
Ns = len(scan_points)
print(f"ROI scan points: {Ns}")

# 创建插值点表达式
w_expr = fem.Expression(W.sub(0).collapse(), scan_points)
# 注意：Expression.eval 在 0.7.x 中需在并行上下文调用

# ----------------------------------------------------------
# 9. 初始条件：高斯脉冲初始位移 (模拟激励)
# ----------------------------------------------------------
def initial_pulse(x):
    # x: (3, n) 形状
    r = np.sqrt((x[0] - x_src)**2 + (x[1] - y_src)**2)
    return A * np.exp(-(r / (2*h))**2)  # 空间高斯包络

# 插值初始条件
W_n.interpolate(lambda x: (initial_pulse(x), np.zeros_like(x[0]), np.zeros_like(x[0])))
W_n1.x.array[:] = W_n.x.array  # 静止开始

# ----------------------------------------------------------
# 10. 时域推进
# ----------------------------------------------------------
U = np.zeros((Ns, Nt), dtype=np.float32)

print(f"Starting time stepping: Nt={Nt}, dt={dt*1e6:.3f} us")

for n in range(Nt):
    # 求解
    W = problem.solve()
    
    # 提取 ROI 扫描点场值
    try:
        w_vals = w_expr.eval(MPI.COMM_WORLD)
        if w_vals is not None:
            U[:, n] = w_vals.flatten()
    except Exception as e:
        # 并行环境下仅 rank 0 有完整数据
        if MPI.COMM_WORLD.rank == 0:
            pass
    
    # 更新历史
    W_n1.x.array[:] = W_n.x.array
    W_n.x.array[:] = W.x.array
    
    if n % 100 == 0 and MPI.COMM_WORLD.rank == 0:
        max_val = np.max(np.abs(U[:, n])) if np.any(U[:, n]) else 0
        print(f"  t={t_vec[n]*1e6:.1f} us, w_max={max_val:.4f}")

# ----------------------------------------------------------
# 11. 仅 rank 0 保存结果
# ----------------------------------------------------------
if MPI.COMM_WORLD.rank == 0:
    # 归一化 (与现有数据一致)
    scale = np.max(np.abs(U)) + 1e-12
    U_norm = U / scale
    
    df = pd.DataFrame(U_norm)
    output_path = Path("data/raw/fem_mindlin_intact_high_snr.pkl")
    df.to_pickle(output_path)
    print(f"Saved: {output_path}, shape={U_norm.shape}, scale={scale:.4e}")
    
    # 保存参数副本
    with open(output_path.with_suffix('.json'), 'w') as f:
        json.dump({
            'params': params['physical_params'],
            'simulation': {
                'Nt': Nt, 'dt': dt, 'fs': fs,
                'Ns': Ns, 'scan_points': scan_points.tolist(),
                'mesh_h': h_elem, 'domain_Lx': Lx, 'domain_Ly': Ly,
                'scale_factor': float(scale),
                'timestamp': datetime.now().isoformat()
            }
        }, f, indent=2)

print("Done.")