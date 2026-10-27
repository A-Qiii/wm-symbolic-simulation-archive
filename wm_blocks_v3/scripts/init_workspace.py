"""Create ignored runtime directories; do not alter existing outputs."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from paths import ROOT,GPU_LOCK
def main():
 for name in ['runs','logs','data','cache','models','reports','engineering']:(ROOT/name).mkdir(exist_ok=True)
 GPU_LOCK.parent.mkdir(parents=True,exist_ok=True);GPU_LOCK.touch(exist_ok=True)
if __name__=='__main__':main()
