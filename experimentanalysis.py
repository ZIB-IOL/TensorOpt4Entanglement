import os

class ExperimentInstance:
    def __init__(self, filename: str, name: str, statetype: str, nsubs: int, dimH: int):
        self.filename = filename
        self.name = name
        self.statetype = statetype
        self.nsubs = nsubs
        self.dimH = dimH

class ExperimentResult:
    def __init__(self, instance: str, algo: str, glbub: float, approxub: float, glblb: float, approxweights: float, time: float):
        self.instance = instance
        self.algo = algo
        self.glbub = glbub
        self.approxub = approxub
        self.glblb = glblb
        self.approxweights = approxweights
        self.time = time


def load_benchmarks(benchmark_dir):
    instances = []
    for filename in os.listdir(benchmark_dir):
        print(f"Processing file: {filename}")
        filepath = os.path.join(benchmark_dir, filename)
        with open(filepath, 'r') as f:
            lines = f.readlines()
            # 搜索文件内容，找到包含'nsubs'和'dimH'的行并提取数值
            nsubs = None
            name = None
            for line in lines:
                if line[0] == 'N':
                    nsubs = int(line.split('=')[1].strip())
                if 'name' in line:
                    name = line.split('=')[1].strip()
            stateype = 'GHZ' if 'GHZ' in name else 'Dicke' if 'Dicke' in name else 'Unknown'
        instances.append(ExperimentInstance(filename, name, stateype, nsubs, 2**nsubs))
    return instances

def load_results(result_dir):
    results = []
    for result_file in os.listdir(results_dir):
        result_path = os.path.join(results_dir, result_file)
        with open(result_path, 'r') as f:
            lines = f.readlines()
            algorithms = {}
            for line in lines:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                if line.startswith('glbub'):
                    glbub = float(line.split(':')[1].strip())
                elif line.startswith('approxub'):
                    approxub = float(line.split(':')[1].strip())
                elif line.startswith('glblb'):
                    glblb = float(line.split(':')[1].strip())
                elif line.startswith('approxweights'):
                    approxweights = float(line.split(':')[1].strip())
                elif line.startswith('time'):
                    time = float(line.split(':')[1].strip())
                elif line.startswith('algo'):
                    algo = line.split(':')[1].strip()
                elif line.startswith('instance'):
                    instance = line.split(':')[1].strip()
            result = ExperimentResult(instance, algo, glbub, approxub, glblb, approxweights, time)
            results.append(result)
    return results

benchmark_dir = os.path.join(os.getcwd(), 'benchmark')
instances = load_benchmarks(benchmark_dir)

results_dir = os.path.join(os.getcwd(), 'results')
results = load_results(results_dir)

algos = ["A", "LD1", "D", "LDL"]
algonames = {
    "A": "Alternating",
    "LD1": "LADMM",
    "D": "Discretization",
    "LDL": "Lifting-Discretization"
}
instance_results = {}

instances.sort(key=lambda inst: inst.nsubs)

for instance in instances:
    instance_results[instance.filename] = []
    for result in results:
        if result.instance == instance.filename:
            instance_results[instance.filename].append(result)
    printoutstr =  "\\midrule \n \\multirow{4}{*}{" + instance.name.replace('"', '').replace('_', '\_') + "} \n"
    for algo in algos:
        found = False
        for result in instance_results[instance.filename]:
            if result.algo == algo:
                printoutstr += f"& {algonames[algo]} & {result.glbub:.6f} & {result.glblb:.6f} & {int(result.time)} \\\\ \n "
                found = True
                break
        if not found:
            printoutstr += " & N/A & N/A & N/A "
    print(printoutstr)
print("\\bottomrule")