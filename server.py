from openreward.environments import Server

from opendeepresearch import OpenDeepResearch

if __name__ == "__main__":
    server = Server([OpenDeepResearch])
    server.run()
