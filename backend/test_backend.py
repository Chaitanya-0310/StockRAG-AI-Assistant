import sys
import os

# Add the parent directory to sys.path to allow importing backend modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from backend.main import app
    from backend.services.stock_service import fetch_stock_history
    from backend.models.models import StockPrice
    print("✅ Imports successful.")
except ImportError as e:
    print(f"❌ Import failed: {e}")
    sys.exit(1)

print("Backend code structure seems correct.")
