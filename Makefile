.PHONY: setup test train optimize serve
setup:
\tpython -m pip install -U pip && pip install -r requirements.txt
test:
\tpytest -q
train:
\tpython -m main demo_train
optimize:
\tpython -m main optimize
serve:
\tpython -m main serve
