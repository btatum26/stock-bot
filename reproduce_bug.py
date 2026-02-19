from src.gui import ChartWindow
from PyQt6.QtWidgets import QApplication
import sys
import os

# Ensure we are in the right directory
os.chdir(os.path.dirname(os.path.abspath(__file__)))

def test_load_chart():
    app = QApplication(sys.argv)
    window = ChartWindow()
    
    print(f"Main plot scene at init: {window.main_plot.scene()}")
    
    try:
        # Simulate clicking load chart
        print("Calling load_chart...")
        window.controls.ticker_input.setText("AAPL")
        window.load_chart()
        print("load_chart called successfully.")
        
        # Check if overlays have items
        print(f"Price overlay items: {len(window.price_overlay.items)}")
        print(f"Volume overlay in scene: {window.volume_overlay.vol_view.scene() is not None}")
        
    except Exception as e:
        print(f"Caught unexpected exception: {e}")
        import traceback
        traceback.print_exc()
    finally:
        app.quit()

if __name__ == "__main__":
    test_load_chart()
