import os
import unittest

os.environ["HOME"] = os.path.realpath(os.path.dirname(__file__))  # For temporary cache folders to be generated in the test folder (and cleaned up)
if os.environ["HOME"].startswith("/home/"):
    os.environ["HOME"] = "/".join(os.environ["HOME"].split("/")[:3])

# root_dir = os.path.dirname(os.path.dirname(__file__))
# sys.path.append(root_dir)

from unittests import *

if __name__ == "__main__":
    unittest.main()
