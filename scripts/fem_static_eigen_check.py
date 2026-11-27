#!/usr/bin/env python3
"""
FEniCSx 静力/特征值验证脚本
- KL 板与 MR 板静力求解对比解析解
- 特征值求解对比理论频率
- 网格收敛性检查
"""
import numpy as np
import dolfinx
import ufl
from dolfinx import fem, mesh, io
from dolfinx.fem import functionspace, dirichletbc, locate_dofs_topological
from dolfinx.fem.petsc import LinearProblem, LinearSolver
from mpi4py import MPI
from petsc4py import PETSc
from slepc4py import SLEPc
import gmsh
import yaml
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ----------------------------------------------------------
# 读取参数
# ----------------------------------------------------------
with open("params/physical_params.yaml") as f:
    params = yaml.safe_load(f)

mat = params['physical_params']['material']
geo = params['physical_params']['geometry']

E = mat['young_modulus_pa']
nu = mat['poisson_ratio']
rho = mat['density_kg_m3']
h = geo['thickness_m']
Lx = geo['plate_length_x_m']
Ly = geo['plate_width_y_m']

D = E * h**3 / (12 * (1 - nu**2))
G = E / (2 * (1 + nu))
kappa = 5/6
kappaGh = kappa * G * h
rho_h = rho * h

# ----------------------------------------------------------
# 解析解 (Navier 解，简支矩形板)
# ----------------------------------------------------------
def kl_static_analytical(q, m_max=10, n_max=10):
    """均布载荷 q 下简支板中心挠度"""
    w_center = 0
    for m in range(1, m_max+1, 2):
        for n in range(1, n_max+1, 2):
            km = m * np.pi / Lx
            kn = n * np.pi / Ly
            k2 = km**2 + kn**2
            w_center += (16 * q / (np.pi**6 * D)) * (1/(m*n)) * (1/k2**2)
    return w_center

def kl_eigen_analytical(m, n):
    """简支板第 (m,n) 阶固有频率"""
    km = m * np.pi / Lx
    kn = n * np.pi / Ly
    omega = np.pi**2 * np.sqrt(D / (rho * h)) * (km**2 + kn**2)**2
    return omega / (2 * np.pi)

def mr_eigen_analytical(m, n):
    """Mindlin 板简支固有频率 (近似)"""
    omega_kl = kl_eigen_analytical(m, n) * 2 * np.pi
    km = m * np.pi / Lx
    kn = n * np.pi / Ly
    k2 = km**2 + kn**2
    # 简化修正因子
    correction = 1 / np.sqrt(1 + (D/(kappaGh)) * k2 + (rho*h**3/12)/(rho_h) * omega_kl**2 / (kappaGh))
    return omega_kl * correction / (2 * np.pi)

# ----------------------------------------------------------
# 生成网格
# ----------------------------------------------------------
def create_mesh(h_elem):
    gmsh.initialize()
    gmsh.model.add("plate")
    gmsh.model.occ.addRectangle(0, 0, 0, Lx, Ly, 1)
    gmsh.model.occ.synchronize()
    gmsh.model.addPhysicalGroup(2, [1], 1, "plate")
    gmsh.model.addPhysicalGroup(1, [1,2,3,4], 2, "boundaries")
    gmsh.model.occ.synchronize()
    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", h_elem)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", h_elem)
    gmsh.model.mesh.generate(2)
    domain, cell_tags, facet_tags = dolfinx.io.gmshio.model_to_mesh(
        gmsh.model, MPI.COMM_WORLD, 0, gdim=2
    )
    gmsh.finalize()
    return domain, facet_tags

# ----------------------------------------------------------
# KL 板静力求解
# ----------------------------------------------------------
def solve_kl_static(domain, facet_tags, h_elem):
    V = functionspace(domain, ("CG", 2))
    w = ufl.TrialFunction(V)
    dw = ufl.TestFunction(V)

    # 简支边界 w=0
    def boundary(x):
        return np.isclose(x[0], 0) | np.isclose(x[0], Lx) | np.isclose(x[1], 0) | np.isclose(x[1], Ly)
    boundary_facets = locate_dofs_topological(V, 1, facet_tags.find(2))
    bc = dirichletbc(fem.Constant(domain, 0.0), boundary_facets, V)

    # 均布载荷
    q = 1.0  # N/m^2
    a = D * ufl.inner(ufl.hess(w), ufl.hess(dw)) * ufl.dx
    L = q * dw * ufl.dx

    problem = LinearProblem(a, L, bcs=[bc], petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
    w_sol = problem.solve()

    # 中心挠度
    center = np.array([Lx/2, Ly/2, 0.0])
    w_expr = fem.Expression(w_sol, center.reshape(1,3))
    w_center = w_expr.eval(MPI.COMM_WORLD)
    return float(w_center[0]) if w_center is not None else 0.0

# ----------------------------------------------------------
# MR 板静力求解 (三场)
# ----------------------------------------------------------
def solve_mr_static(domain, facet_tags, h_elem):
    V_w = functionspace(domain, ("CG", 2))
    V_tx = functionspace(domain, ("CG", 1))
    V_ty = functionspace(domain, ("CG", 1))
    from dolfinx.fem import MixedFunctionSpace
    V = MixedFunctionSpace([V_w, V_tx, V_ty])

    w, tx, ty = ufl.TrialFunctions(V)
    dw, dtx, dty = ufl.TestFunctions(V)

    # 简支边界
    boundary_facets = locate_dofs_topological(V_w, 1, facet_tags.find(2))
    bcs = [
        dirichletbc(fem.Constant(domain, 0.0), boundary_facets, V_w),
        dirichletbc(fem.Constant(domain, 0.0), boundary_facets, V_tx),
        dirichletbc(fem.Constant(domain, 0.0), boundary_facets, V_ty),
    ]

    kappa_xx = -tx.dx(0)
    kappa_yy = -ty.dx(1)
    kappa_xy = -0.5 * (tx.dx(1) + ty.dx(0))
    gamma_xz = tx + w.dx(0)
    gamma_yz = ty + w.dx(1)

    Mxx = D * (kappa_xx + nu * kappa_yy)
    Myy = D * (kappa_yy + nu * kappa_xx)
    Mxy = D * (1 - nu) * kappa_xy
    Qx = kappaGh * gamma_xz
    Qy = kappaGh * gamma_yz

    q = 1.0
    a = (Mxx * (-dtx.dx(0)) + Myy * (-dty.dx(1)) + Mxy * (-0.5*(dtx.dx(1) + dty.dx(0)))
         + Qx * (dtx + dw.dx(0)) + Qy * (dty + dw.dx(1))) * ufl.dx
    L = q * dw * ufl.dx

    problem = LinearProblem(a, L, bcs=bcs, petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
    W_sol = problem.solve()

    w_sol = W_sol.sub(0).collapse()
    center = np.array([Lx/2, Ly/2, 0.0])
    w_expr = fem.Expression(w_sol, center.reshape(1,3))
    w_center = w_expr.eval(MPI.COMM_WORLD)
    return float(w_center[0]) if w_center is not None else 0.0

# ----------------------------------------------------------
# 特征值求解 (KL)
# ----------------------------------------------------------
def solve_kl_eigen(domain, facet_tags, n_eigs=10):
    V = functionspace(domain, ("CG", 2))
    w = ufl.TrialFunction(V)
    dw = ufl.TestFunction(V)

    boundary_facets = locate_dofs_topological(V, 1, facet_tags.find(2))
    bc = dirichletbc(fem.Constant(domain, 0.0), boundary_facets, V)

    a = D * ufl.inner(ufl.hess(w), ufl.hess(dw)) * ufl.dx
    m = rho_h * w * dw * ufl.dx

    A = fem.petsc.assemble_matrix(fem.form(a), bcs=[bc])
    A.assemble()
    M = fem.petsc.assemble_matrix(fem.form(m), bcs=[bc])
    M.assemble()

    eps = SLEPc.EPS().create()
    eps.setOperators(A, M)
    eps.setProblemType(SLEPc.EPS.ProblemType.GHEP)
    eps.setDimensions(n_eigs, PETSc.DECIDE, PETSc.DECIDE)
    eps.setWhichEigenpairs(SLEPc.EPS.Which.SMALLEST_REAL)
    eps.solve()

    nconv = eps.getConverged()
    eigvals = []
    for i in range(min(nconv, n_eigs)):
        eigval = eps.getEigenvalue(i)
        eigvals.append(float(eigval.real))
    return eigvals

# ----------------------------------------------------------
# 主流程
# ----------------------------------------------------------
def main():
    print("=== FEniCSx Static & Eigen Validation ===")
    print(f"Plate: {Lx*1000:.1f} x {Ly*1000:.1f} mm, h={h*1000:.2f} mm")
    print(f"Material: E={E/1e9:.1f} GPa, nu={nu}, rho={rho} kg/m3")

    # 网格收敛研究
    h_elems = [0.5e-3, 0.25e-3, 0.125e-3]  # 粗->细
    results = {
        'mesh_sizes': [float(h*1000) for h in h_elems],
        'kl_static': [],
        'mr_static': [],
        'kl_eigen': [],
        'analytical': {
            'kl_static': kl_static_analytical(1.0),
            'kl_eigen': [kl_eigen_analytical(m, n) for m in range(1,4) for n in range(1,4)],
            'mr_eigen': [mr_eigen_analytical(m, n) for m in range(1,4) for n in range(1,4)],
        }
    }

    for h_elem in h_elems:
        print(f"\n--- Mesh size: {h_elem*1000:.3f} mm ---")
        domain, facet_tags = create_mesh(h_elem)

        # KL 静力
        w_kl = solve_kl_static(domain, facet_tags, h_elem)
        results['kl_static'].append(w_kl)
        print(f"  KL center deflection: {w_kl*1e6:.4f} um")

        # MR 静力
        w_mr = solve_mr_static(domain, facet_tags, h_elem)
        results['mr_static'].append(w_mr)
        print(f"  MR center deflection: {w_mr*1e6:.4f} um")

        # KL 特征值
        eigvals = solve_kl_eigen(domain, facet_tags)
        results['kl_eigen'].append(eigvals)
        print(f"  KL eigenfrequencies (Hz): {[f'{f:.1f}' for f in eigvals[:5]]}")

    # 解析解
    w_anal = results['analytical']['kl_static']
    print(f"\nAnalytical KL center deflection: {w_anal*1e6:.4f} um")
    print(f"KL error (finest): {abs(results['kl_static'][-1] - w_anal)/w_anal*100:.2f}%")
    print(f"MR error (finest): {abs(results['mr_static'][-1] - w_anal)/w_anal*100:.2f}%")
    print(f"KL/MR difference (finest): {abs(results['kl_static'][-1] - results['mr_static'][-1])/w_anal*100:.2f}%")

    # 保存结果
    output = {
        'params': params['physical_params'],
        'mesh_convergence': results,
        'summary': {
            'kl_static_error_finest_percent': abs(results['kl_static'][-1] - w_anal)/w_anal*100,
            'mr_static_error_finest_percent': abs(results['mr_static'][-1] - w_anal)/w_anal*100,
            'kl_mr_diff_percent': abs(results['kl_static'][-1] - results['mr_static'][-1])/w_anal*100,
        }
    }

    Path("data/raw").mkdir(parents=True, exist_ok=True)
    Path("analysis").mkdir(parents=True, exist_ok=True)

    with open("data/raw/fem_static_eigen_results.json", "w") as f:
        json.dump(output, f, indent=2, default=str)

    with open("analysis/P1-2_3_static_eigen_validation.md", "w") as f:
        f.write("# KL/MR 静力与特征值验证报告\n\n")
        f.write(f"**Plate**: {Lx*1000:.1f} x {Ly*1000:.1f} mm, h={h*1000:.2f} mm\n\n")
        f.write(f"**Material**: E={E/1e9:.1f} GPa, ν={nu}, ρ={rho} kg/m³\n\n")
        f.write("## 网格收敛 - 中心挠度 (均布载荷 q=1 N/m²)\n\n")
        f.write("| h (mm) | KL (μm) | MR (μm) | 解析解 (μm) | KL误差 | MR误差 | KL/MR差异 |\n")
        f.write("|--------|---------|---------|-------------|--------|--------|-----------|\n")
        for i, h_elem in enumerate(h_elems):
            f.write(f"| {h_elems[i]*1000:.3f} | {results['kl_static'][i]*1e6:.4f} | {results['mr_static'][i]*1e6:.4f} | "
                    f"{w_anal*1e6:.4f} | {abs(results['kl_static'][i]-w_anal)/w_anal*100:.2f}% | "
                    f"{abs(results['mr_static'][i]-w_anal)/w_anal*100:.2f}% | "
                    f"{abs(results['kl_static'][i]-results['mr_static'][i])/w_anal*100:.2f}% |\n")

        f.write("\n## 特征值对比 (前5阶)\n\n")
        f.write("| 阶次 | 解析解 (Hz) | 数值解 (Hz, 最细网格) | 相对误差 |\n")
        f.write("|------|-------------|----------------------|----------|\n")
        eig_fine = results['kl_eigen'][-1]
        for i, f_anal in enumerate(results['analytical']['kl_eigen'][:5]):
            if i < len(eig_fine):
                err = abs(eig_fine[i] - f_anal) / f_anal * 100
                f.write(f"| {i+1} | {f_anal:.2f} | {eig_fine[i]:.2f} | {err:.2f}% |\n")

        f.write(f"\n## 结论\n\n")
        f.write(f"- KL 静力最细网格误差: **{output['summary']['kl_static_error_finest_percent']:.2f}%** {'✅ < 2%' if output['summary']['kl_static_error_finest_percent'] < 2 else '❌ > 2%'}\n")
        f.write(f"- MR 静力最细网格误差: **{output['summary']['mr_static_error_finest_percent']:.2f}%** {'✅ < 5%' if output['summary']['mr_static_error_finest_percent'] < 5 else '❌ > 5%'}\n")
        f.write(f"- KL/MR 差异: **{output['summary']['kl_mr_diff_percent']:.2f}%** {'✅ < 3%' if output['summary']['kl_mr_diff_percent'] < 3 else '❌ > 3%'}\n")
        f.write(f"- 薄板极限下 MR→KL 收敛: {'是' if output['summary']['kl_mr_diff_percent'] < 3 else '否'}\n")

    print("\n=== Validation completed ===")
    print(f"Results saved to: data/raw/fem_static_eigen_results.json")
    print(f"Report saved to: analysis/P1-2_3_static_eigen_validation.md")


if __name__ == "__main__":
    main()