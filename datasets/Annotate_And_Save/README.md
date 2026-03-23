# NautilusVision
To use annotate_labels, it is simple. 

Start by looking at the top lines of the code and make sure you are annotating the correct dataset by looking at the relative path.

If it is okay, simply run the code. You should see an image with the gui pop up on your screen. The controls are simple:
    - You can zoom in or out with your scrool wheel, the scrool factor is shown at the top of the screen
    - You can save the labels by pressing "s" on your keyboard
    - You can save the labels and go to the next image by pressing "n"
    - You can delete the closest bounding box by pressing "d" while hovering your mouse inside of it
    - You can draw a bounding box by clicking with your mouse. You should see a blue rectangle appear and change size when you drag your mouse. Get it to the size you like and click again. You can cancel by pressing "c" if desired
    - You can swap the type of object you are anotating by pressing the numbres "1" to "9" on top of your keyboard. The current class selected is shown at the top of the screen.
    - You will see at the bottom of the screen what image you are at. If you want to stop, simply press "esc", it will close everything
    - If you want to start annotating from a specific image, you can change the START_INDEX at the top of the code near the paths before starting. 
