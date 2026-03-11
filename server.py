from openreward.environments import Server

from dsbc import DSBC

if __name__ == "__main__":
    server = Server([DSBC])
    server.run()
