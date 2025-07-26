from qiskit import QuantumCircuit, Aer, execute

qc = QuantumCircuit(2)
qc.h(0)
qc.cx(0, 1)
qc.measure_all()

backend = Aer.get_backend('qasm_simulator')

job = execute(qc, backend, shots=1024, backend_options={"max_parallel_threads": 40})
result = job.result()
print(result.get_counts())

