source "C:/working/workspaces/WSU/wsu_venv/Scripts/activate"
rm ./wheels/rarnold_cs6840_final_project-0.0.0-py3-none-any.whl
python setup.py bdist_wheel -d ./wheels
pip uninstall rarnold-cs6840-final-project
pip install ./wheels/rarnold_cs6840_final_project-0.0.0-py3-none-any.whl
