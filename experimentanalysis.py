import os
import math
class ExperimentInstance:
    def __init__(self, filename: str, name: str, statetype: str, nsubs: int, dimH: int):
        self.filename = filename
        self.name = name
        self.statetype = statetype
        self.nsubs = nsubs
        self.dimH = dimH

class ExperimentResult:
    def __init__(self, instance: str, algo: str, glbub: float, approxub: float, glblb: float, approxweights: float, approxfeas: float, time: float, result_path):
        self.instance = instance
        self.algo = algo
        self.glbub = glbub
        self.approxub = approxub
        self.glblb = glblb
        self.approxweights = approxweights
        self.approxfeas = approxfeas
        self.time = time
        self.result_path = result_path


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
        # skip the 'outputs' folder and any directories
        if result_file == 'outputs':
            continue
        if os.path.isdir(os.path.join(results_dir, result_file)):
            continue
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
                elif line.startswith('approxfeas'):
                    approxfeas = float(line.split(':')[1].strip())
                elif line.startswith('time'):
                    time = float(line.split(':')[1].strip())
                elif line.startswith('algo'):
                    algo = line.split(':')[1].strip()
                elif line.startswith('instance'):
                    instance = line.split(':')[1].strip()
            result = ExperimentResult(instance, algo, glbub, approxub, glblb, approxweights, approxfeas, time, result_path)
            results.append(result)
    return results

benchmark_dir = os.path.join(os.getcwd(), 'benchmark')
instances = load_benchmarks(benchmark_dir)

results_dir = os.path.join(os.getcwd(), 'results')
results = load_results(results_dir)

algos = ["A", "LD1", "D", "LDL", "PPT"]
algonames = {
    "A": "Alt-SDP",
    "LD1": "LADMM",
    "D": "CP",
    "LDL": "IR",
    "PPT": "DPS"
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
                glbub_str = '-' if result.glbub == 0.0 else f'{result.glbub:.5f}'
                glblb_str = '-' if not math.isfinite(result.glblb)  else f'{result.glblb:.5f}'
                approxub_str = '-' if result.approxfeas == 0.0 else f'{result.approxub:.5f}'
                approxfeas_str = '-' if result.approxfeas == 0.0 else f'{result.approxfeas:.5f}'
                printoutstr += f"& {algonames[algo]} & {glbub_str} & {glblb_str} & {approxub_str} & {approxfeas_str} & {int(result.time)} \\\\ \n "
                found = True
                break
        if not found:
            printoutstr += " & N/A & N/A & N/A & N/A & N/A \\\\ \n"
    print(printoutstr)
print("\\bottomrule")
