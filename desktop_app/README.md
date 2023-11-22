# Instructions for repackaging the app (its now called Assurance.py, and I've made the internal run a little more consistent)

## Step 0- Important!
Move your current dist file out of the web_app directory and even out of the assurance directory (just so that you have the old version of the app saved)

## Step 1:
Go into the assurance directory (where the GA and everything is hosted), and. Run `git pull`

## Step 2: Go into desktop_app
Go into desktop_app and run `pyinstaller --add-data 'templates;templates' --add-data 'static;static' --hidden-import=numpy --hidden-import=graphviz -w Assurance.py`

## Step 3:
Once the run is complete, open up your file explorer, and navigate back to web_app, and you should see a directory called dist. In dist, you'll see the executable (wherever it was earlier, but now labeled Assurance), with a folder next to it. Click on the executable, and let it run. 

# Note: If you want to change something, and repackage the executable (Instructions vary at step 2 and 3)

## Step 1:
Make your change, either in the appropriate templates folder, or in Assurance.py

## Step 2:
Run `rm -rf build`

## Step 3:
Run `pyinstaller Assurance.spec`. As pyinstaller runs, it will ask you for permission a couple of times, type `y` each time, then enter.

## Step 4:
Once the run is complete, open up your file explorer, and navigate back to web_app, and you should see a directory called dist. In dist, you'll see the executable (wherever it was earlier, but now labeled Assurance), with a folder next to it. Click on the executable, and let it run. 