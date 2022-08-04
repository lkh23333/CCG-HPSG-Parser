## evaluation of the parsing part
```
cd py_parsing/evaluation
export CANDC=candc
python -m depccg.tools.evaluate ../../data/wsj_00.parg ../../data/wsj_00.predicted.auto
```