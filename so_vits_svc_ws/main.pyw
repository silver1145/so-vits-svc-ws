import os
import sys

nullfile = open(os.devnull, "w")
sys.stdout = nullfile
sys.stderr = nullfile

from main import main

main()
os._exit(0)
