py -m virtualenv .env &  if not exist ".env\data" mkdir .env\data & if not exist ".env\data\videos" mkdir .env\data\videos & if not exist ".env\data\gazedata" mkdir .env\data\gazedata & .env\Scripts\activate && py -m pip install -r requirements.txt
pause
