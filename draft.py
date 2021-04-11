from tqdm import tqdm
training_batch = [
    ["mot", "hai"],
    ["3", "4"],
    ["5", "6"],
]

for a, b  in tqdm(enumerate(training_batch)):
    print("a : ", a)
    print("b : ", b)
