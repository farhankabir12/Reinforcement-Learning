import json
import matplotlib.pyplot as plt
from pathlib import Path


with open('scripts/results/MPE/simple_spread/mappo/check/run2/logs/summary.json', 'r') as openfile:

    data = json.load(openfile)
    x = []
    y = []
    bagagge = "/home/cail/Desktop/farhan_brotee/MAPPO/Official MAPPO/on-policy-official/onpolicy/scripts/results/MPE/simple_spread/mappo/check/run2/logs/"
    ag0 = "agent0/individual_rewards/agent0/individual_rewards"
    ag1 = "agent1/individual_rewards/agent1/individual_rewards"

    j = 0
    print(data[bagagge + ag1])
    for i in data[bagagge + ag0]:
        y.append(i[2])
        x.append(j)
        j += 1

    plt.plot(x, y)
    plt.show()
