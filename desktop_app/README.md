# Instructions I have so far for packaging the app for windows (for repackaging, skip step 2)

## Step 1:
Go into the assurance directory (where the GA and everything is hosted), and then go into the web_app directory. Run `git pull`

## Step 2:
Run `pip3 install pyinstaller` and `pip3 install flaskwebgui`

## Step 3:
Run `pyinstaller --add-data 'templates;templates' --add-data 'static;static' --hidden-import=numpy --hidden-import=graphviz -w app.py`

## Step 4:
Once the run is complete, open up your file explorer, and navigate back to web_app, and you should see a directory called dist. In dist, you'll see the executable, with a folder next to it. Click on the executable, and let it run. 