# Import các thư viện cần thiết
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp
from qiskit_algorithms import VQE
from qiskit_algorithms.optimizers import COBYLA
from qiskit_aer import AerSimulator
from qiskit.circuit.library import RealAmplitudes
import numpy as np

# Bước 1: Định nghĩa Hamiltonian
hamiltonian = SparsePauliOp.from_list([("Z", 1.0)])

# Bước 2: Tạo ansatz
ansatz = RealAmplitudes(num_qubits=1, reps=2)

# Bước 3: Thiết lập optimizer
optimizer = COBYLA(maxiter=100)

# Bước 4: Thiết lập backend
backend = AerSimulator()

# Bước 5: Khởi tạo VQE
vqe = VQE(ansatz=ansatz, optimizer=optimizer, quantum_instance=backend)

# Bước 6: Chạy VQE
result = vqe.compute_minimum_eigenvalue(operator=hamiltonian)

# Bước 7: In kết quả
print("Năng lượng trạng thái cơ bản:", result.eigenvalue.real)
print("Tham số tối ưu của ansatz:", result.optimal_parameters)
print("Số lần đánh giá hàm mục tiêu:", result.optimizer_evals)
