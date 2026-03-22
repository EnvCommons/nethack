from openreward.environments import Server

from nethack_env import NetHackEnvironment

if __name__ == "__main__":
    server = Server([NetHackEnvironment])
    server.run()
