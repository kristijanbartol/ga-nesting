Install newton:

```
uninstall newton pip uninstall newton (or something like python3.11 -m pip uninstall newton)
git clone https://github.com/newton-physics/newton.git
cd newton
git checkout e60785bc96d3678ed5b1b99ede507b27a17896c0 (you will in a "detached head" state, that's expected)
install this newton pip install -e .
run original simulation.py
```
