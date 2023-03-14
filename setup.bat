py -m virtualenv venv &  if not exist "venv\data" mkdir venv\data & if not exist "venv\data\videos" mkdir venv\data\videos & if not exist "venv\data\gazedata" mkdir venv\data\gazedata & venv\Scripts\activate && py -m pip install -r requirements.txt
pause
