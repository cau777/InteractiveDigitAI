cd ..
call .\Python\venv\Scripts\activate.bat
python -m build .\Python
xcopy /I /Y /F /Q .\Python\dist\*.whl .\Client\src\assets\python\
xcopy /I /Y /F /Q .\Python\client_scripts\*.py .\Client\src\assets\python\
