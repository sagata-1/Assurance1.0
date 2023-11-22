# Instructions for repackaging app (its now called Assurance.py)

## Step 1:
Go into the assurance directory (where the GA and everything is hosted), and then go into the web_app directory. Run `git pull`

## Step 2 (Only do if you see the build folder in the directory)
Run `rm -rf build`

## Step 3:
Run `pyinstaller --add-data 'templates;templates' --add-data 'static;static' --hidden-import=numpy --hidden-import=graphviz -w Assurance.py`

## Step 4:
Once the run is complete, open up your file explorer, and navigate back to web_app, and you should see a directory called dist. In dist, you'll see the executable (wherever it was earlier, but now labeled Assurance), with a folder next to it. Click on the executable, and let it run. 

# Note: If you want to change something, and repackage the executable (Instructions vary at step 3)

## Step 1:
Make your change, either in the appropriate templates folder, or in Assurance.py

## Step 2:
Run `rm -rf build`

## Step 3:
Run `pyinstaller Assurance.spec`. As pyinstaller runs, it will ask you for permission a couple of times, type `y` each time, then enter.

## Step 4:
Once the run is complete, open up your file explorer, and navigate back to web_app, and you should see a directory called dist. In dist, you'll see the executable (wherever it was earlier, but now labeled Assurance), with a folder next to it. Click on the executable, and let it run. 