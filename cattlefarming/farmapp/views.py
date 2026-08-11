# # farmapp/views.py
# from django.shortcuts import render, redirect
# from .forms import FarmerInputForm
# from .ml_utils import predict_from_input, CSV_PATH, FEATURE_COLS
# import pandas as pd
# from django.views.decorators.csrf import csrf_exempt
# from datetime import datetime
# import threading

# file_lock = threading.Lock()

# # def predict_view(request):
# #     if request.method == 'POST':
# #         form = FarmerInputForm(request.POST)
# #         if form.is_valid():
# #             data = {k: form.cleaned_data[k] for k in FEATURE_COLS}
# #             # make prediction
# #             preds = predict_from_input(data)
# #             # show prediction and provide button/form to submit actuals (or save predicted row)
# #             context = {'form': form, 'preds': preds}
# #             return render(request, 'farmapp/predict.html', context)
# #     else:
# #         form = FarmerInputForm()
# #     return render(request, 'farmapp/predict.html', {'form': form})

# # def predict_view(request):
# #     if request.method == 'POST':
# #         form = FarmerInputForm(request.POST)
# #         if form.is_valid():
# #             data = {k: form.cleaned_data[k] for k in FEATURE_COLS}
# #             preds = predict_from_input(data)
# #             return render(request, 'farmapp/predict.html', {
# #                 'form': form,
# #                 'preds': preds   # <- yeh zaroor hona chahiye
# #             })
# #     else:
# #         form = FarmerInputForm()
# #     return render(request, 'farmapp/predict.html', {'form': form})


# def predict_view(request):
#     if request.method == 'POST':
#         form = FarmerInputForm(request.POST)
#         if form.is_valid():
#             data = {k: form.cleaned_data[k] for k in FEATURE_COLS}
#             preds = predict_from_input(data)

#             # CSV me directly save karo (date + features + preds)
#             now = datetime.now().strftime('%d-%m-%Y %H:%M')
#             row = {'date': now}
#             for f in FEATURE_COLS:
#                 row[f] = data[f]
#             row['milk_liters'] = preds['milk_liters']
#             row['disease_label'] = preds['disease_label']
#             append_row_to_csv(row)

#             return render(request, 'farmapp/predict.html', {
#                 'form': form,
#                 'preds': preds
#             })
#     else:
#         form = FarmerInputForm()
#     return render(request, 'farmapp/predict.html', {'form': form})


# def append_row_to_csv(row_dict):
#     # ensure columns order matches CSV
#     cols = ['date'] + FEATURE_COLS + ['milk_liters','disease_label']
#     df_row = pd.DataFrame([row_dict], columns=cols)
#     header = not pd.io.common.file_exists(CSV_PATH)
#     with file_lock:
#         df_row.to_csv(CSV_PATH, mode='a', header=header, index=False)

# @csrf_exempt
# def submit_actuals(request):
#     # called when farmer confirms/provides actual values
#     if request.method == 'POST':
#         form = FarmerInputForm(request.POST)
#         if form.is_valid():
#             # prepare row to append: date + features + milk_liters + disease_label (prefer actuals if provided)
#             now = datetime.now().strftime('%d-%m-%Y %H:%M')
#             row = {'date': now}
#             for f in FEATURE_COLS:
#                 row[f] = form.cleaned_data[f]
#             # if farmer provided actuals, use them; else make prediction and record that
#             actual_milk = form.cleaned_data.get('actual_milk_liters')
#             actual_disease = form.cleaned_data.get('actual_disease_label')
#             if actual_milk is not None and actual_disease is not None:
#                 row['milk_liters'] = actual_milk
#                 row['disease_label'] = actual_disease
#             else:
#                 # predict and save predicted values (so dataset grows)
#                 preds = predict_from_input({k: row[k] for k in FEATURE_COLS})
#                 row['milk_liters'] = preds['milk_liters']
#                 row['disease_label'] = preds['disease_label']
#             append_row_to_csv(row)
#             # optionally retrain models so next prediction uses updated CSV
#             # from .ml_utils import train_models
#             # train_models(force_retrain=True)
#             return render(request, 'farmapp/submitted.html', {'row': row})
#     return redirect('farmapp:predict')


# # import pandas as pd
# # from django.shortcuts import render
# # from django.conf import settings
# # import os
# # from datetime import timedelta

# # def dashboard(request):
# #     csv_file_path = os.path.join(settings.BASE_DIR, 'data', 'daily_record.csv')

# #     try:
# #         # Load data from the CSV file using pandas
# #         data = pd.read_csv(csv_file_path)

# #         # Standardize column names (strip spaces, lower case)
# #         data.columns = [col.strip().replace(' ', '_').lower() for col in data.columns]

# #         # Convert the dataframe to a list of dictionaries (for use in the template)
# #         cattle_records = data.to_dict(orient='records')

# #         # Calculate KPIs
# #         total_cattle = len(cattle_records)
# #         daily_milk_production = sum(float(record.get('milk_l', 0)) for record in cattle_records if record.get('milk_l'))
# #         health_index = 92  # Placeholder

# #         kpis = [
# #             {'title': 'Total Cattle', 'value': total_cattle, 'icon': 'fas fa-cow', 'icon_bg_color': '#3498db', 'trend': 'trend-up', 'trend_direction': 'up', 'trend_percent': '12%'},
# #             {'title': 'Daily Milk Production', 'value': f'{daily_milk_production} L', 'icon': 'fas fa-tint', 'icon_bg_color': '#2ecc71', 'trend': 'trend-up', 'trend_direction': 'up', 'trend_percent': '5%'},
# #             {'title': 'Health Index', 'value': f'{health_index}%', 'icon': 'fas fa-heartbeat', 'icon_bg_color': '#f1c40f', 'trend': 'trend-down', 'trend_direction': 'down', 'trend_percent': '3%'},
# #             {'title': 'Feed Consumption', 'value': '14.3 kg', 'icon': 'fas fa-utensils', 'icon_bg_color': '#9b59b6', 'trend': 'trend-up', 'trend_direction': 'up', 'trend_percent': '2%'}
# #         ]

# #         # Prepare chart data
# #         data['date'] = pd.to_datetime(data['date'])
# #         recent_dates = data['date'].max() - timedelta(days=7)
# #         recent_data = data[data['date'] >= recent_dates]
# #         milk_trend = recent_data.groupby('date')['milk_l'].sum().reset_index()
# #         milk_trend_labels = milk_trend['date'].dt.strftime('%Y-%m-%d').tolist()
# #         milk_trend_data = milk_trend['milk_l'].tolist()

# #         breed_counts = data['breed'].value_counts()
# #         breed_labels = breed_counts.index.tolist()
# #         breed_data = breed_counts.values.tolist()

# #         context = {
# #             'kpis': kpis,
# #             'cattle_records': cattle_records,
# #             'milk_trend_labels': milk_trend_labels,
# #             'milk_trend_data': milk_trend_data,
# #             'breed_labels': breed_labels,
# #             'breed_data': breed_data,
# #         }

# #     except Exception as e:
# #         print(f"Error reading CSV file: {e}")
# #         context = {
# #             'kpis': [],
# #             'cattle_records': [],
# #             'milk_trend_labels': [],
# #             'milk_trend_data': [],
# #             'breed_labels': [],
# #             'breed_data': [],
# #         }

# #     return render(request, 'farmapp/dashboard.html', context)



# import pandas as pd
# from django.shortcuts import render
# from django.conf import settings
# from django.http import JsonResponse
# import os
# from datetime import timedelta

# def dashboard(request):
#     """Renders the dashboard page."""
#     return render(request, "farmapp/dashboard.html")


# def dashboard_data(request):
#     """Returns dashboard data as JSON for the frontend."""
#     csv_file_path = os.path.join(settings.BASE_DIR, "data", "daily_record.csv")

#     try:
#         data = pd.read_csv(csv_file_path)
#         data.columns = [col.strip().replace(" ", "_").lower() for col in data.columns]
#         data["date"] = pd.to_datetime(data["date"])

#         # Convert to dict
#         cattle_records = data.to_dict(orient="records")

#         # KPIs
#         total_cattle = len(cattle_records)
#         daily_milk_production = sum(float(record.get("milk_l", 0)) for record in cattle_records if record.get("milk_l"))
#         health_index = 92  # Placeholder

#         kpis = {
#             "total_cattle": total_cattle,
#             "total_milk": daily_milk_production,
#             "health_index": health_index,
#             "avg_feed": round(data["feed_kg"].mean(), 2) if "feed_kg" in data.columns else 0,
#         }

#         # Milk trend (last 7 days)
#         recent_dates = data["date"].max() - timedelta(days=7)
#         recent_data = data[data["date"] >= recent_dates]
#         milk_trend = recent_data.groupby("date")["milk_l"].sum().reset_index()

#         milk_trend_labels = milk_trend["date"].dt.strftime("%Y-%m-%d").tolist()
#         milk_trend_data = milk_trend["milk_l"].tolist()

#         # Breed distribution
#         breed_counts = data["breed"].value_counts()
#         breed_labels = breed_counts.index.tolist()
#         breed_data = breed_counts.values.tolist()

#         return JsonResponse({
#             "kpis": kpis,
#             "cattle_records": cattle_records,
#             "milk_trend_labels": milk_trend_labels,
#             "milk_trend_data": milk_trend_data,
#             "breed_labels": breed_labels,
#             "breed_data": breed_data,
#         })

#     except Exception as e:
#         return JsonResponse({"error": str(e)}, status=500)


# farmapp/views.py
from django.shortcuts import render, redirect
from .forms import FarmerInputForm
from .ml_utils import predict_from_input, CSV_PATH, FEATURE_COLS
import pandas as pd
from django.views.decorators.csrf import csrf_exempt
from django.http import JsonResponse
from django.conf import settings
from datetime import datetime, timedelta
import threading
import os
import logging


from django.shortcuts import render
from functools import wraps

def custom_login_required(view_func):
    @wraps(view_func)
    def wrapper(request, *args, **kwargs):
        user_id = request.session.get('user_id')
        if not user_id:
            messages.error(request, "Please log in to access this page.")
            return redirect('farmapp:login_page')
        try:
            user = KeyNest.objects.get(id=user_id)
        except KeyNest.DoesNotExist:
            messages.error(request, "User does not exist. Please log in again.")
            return redirect('farmapp:login_page')
        request.custom_user = user
        return view_func(request, *args, **kwargs)
    return wrapper


def login_page(request):
    
    return render(request, "farmapp/login.html")

# views.py
from django.shortcuts import render, redirect
from django.contrib import messages
from .models import KeyNest

def login_view(request):
    return render(request, "farmapp/login.html")  
from django.shortcuts import render, redirect
from django.contrib import messages
from .models import KeyNest

def login_process(request):
    if request.method == "POST":
        email = request.POST.get('loginEmail')
        password = request.POST.get('loginPassword')
        try:
            user = KeyNest.objects.get(email=email)
            if user.password == password:
                request.session['user_id'] = user.id
                messages.success(request, "Logged in successfully!")
                if user.role == 'Admin':
                    return redirect('farmapp:dashboard')
                return redirect('farmapp:predict')
            else:
                messages.error(request, "Incorrect password.")
        except KeyNest.DoesNotExist:
            messages.error(request, "User not found.")
    return redirect('login_page')


def register_view(request):
    if request.method == "POST":
        name = request.POST.get('registerName')
        email = request.POST.get('registerEmail')
        password = request.POST.get('registerPassword')
        confirm_password = request.POST.get('registerConfirmPassword')

        if password != confirm_password:
            messages.error(request, "Passwords do not match.")
            return redirect('farmapp:login_page')

        if KeyNest.objects.filter(email=email).exists():
            messages.error(request, "Email already registered.")
            return redirect('farmapp:login_page')

        user = KeyNest(name=name, email=email, password=password)
        user.save()
        messages.success(request, "Registration successful! You can log in now.")
        return redirect('farmapp:login_page')

    return render(request, "farmapp/login.html")  

def logout_view(request):
    request.session.flush()
    messages.info(request, "Logged out successfully.")
    return redirect('farmapp:login_page')



# Setup logging
logger = logging.getLogger(__name__)
file_lock = threading.Lock()



@custom_login_required
def predict_view(request):
    

    if request.method == 'POST':
        form = FarmerInputForm(request.POST)
        if form.is_valid():
            data = {k: form.cleaned_data[k] for k in FEATURE_COLS}
            preds = predict_from_input(data)

            # CSV me directly save karo (date + features + preds)
            now = datetime.now().strftime('%d-%m-%Y %H:%M')
            row = {'date': now}
            for f in FEATURE_COLS:
                row[f] = data[f]
            row['milk_liters'] = preds['milk_liters']
            row['disease_label'] = preds['disease_label']
            append_row_to_csv(row)

            return render(request, 'farmapp/predict.html', {
                'form': form,
                'preds': preds
            })
    else:
        form = FarmerInputForm()
    return render(request, 'farmapp/predict.html', {'form': form})


def append_row_to_csv(row_dict):
    # ensure columns order matches CSV
    cols = ['date'] + FEATURE_COLS + ['milk_liters','disease_label']
    df_row = pd.DataFrame([row_dict], columns=cols)
    header = not pd.io.common.file_exists(CSV_PATH)
    with file_lock:
        df_row.to_csv(CSV_PATH, mode='a', header=header, index=False)

@csrf_exempt
def submit_actuals(request):
    # called when farmer confirms/provides actual values
    if request.method == 'POST':
        form = FarmerInputForm(request.POST)
        if form.is_valid():
            # prepare row to append: date + features + milk_liters + disease_label (prefer actuals if provided)
            now = datetime.now().strftime('%d-%m-%Y %H:%M')
            row = {'date': now}
            for f in FEATURE_COLS:
                row[f] = form.cleaned_data[f]
            # if farmer provided actuals, use them; else make prediction and record that
            actual_milk = form.cleaned_data.get('actual_milk_liters')
            actual_disease = form.cleaned_data.get('actual_disease_label')
            if actual_milk is not None and actual_disease is not None:
                row['milk_liters'] = actual_milk
                row['disease_label'] = actual_disease
            else:
                # predict and save predicted values (so dataset grows)
                preds = predict_from_input({k: row[k] for k in FEATURE_COLS})
                row['milk_liters'] = preds['milk_liters']
                row['disease_label'] = preds['disease_label']
            append_row_to_csv(row)
            # optionally retrain models so next prediction uses updated CSV
            # from .ml_utils import train_models
            # train_models(force_retrain=True)
            return render(request, 'farmapp/submitted.html', {'row': row})
    return redirect('farmapp:predict')

@custom_login_required
def dashboard(request):
    """Renders the dashboard page."""
    return render(request, "farmapp/dashboard.html")


# views.py (updated)
def dashboard_data(request):
    """Returns dashboard data as JSON for the frontend."""
    
    print("=== DEBUG: dashboard_data function called ===")
    
    try:
        # Import CSV_PATH safely
        try:
            from .ml_utils import CSV_PATH
            print(f"DEBUG: CSV_PATH from ml_utils: {CSV_PATH}")
        except ImportError as e:
            print(f"DEBUG: Could not import CSV_PATH: {e}")
            CSV_PATH = None
        
        # Multiple possible CSV file paths
        possible_paths = [
            os.path.join(settings.BASE_DIR, "data", "daily_record.csv"),
            os.path.join(settings.BASE_DIR, "daily_record.csv"),
        ]
        
        if CSV_PATH:
            possible_paths.append(CSV_PATH)
        
        print(f"DEBUG: Checking paths: {possible_paths}")
        
        csv_file_path = None
        for path in possible_paths:
            print(f"DEBUG: Checking path: {path}")
            if os.path.exists(path):
                csv_file_path = path
                print(f"DEBUG: Found CSV file at: {path}")
                break
            else:
                print(f"DEBUG: Path does not exist: {path}")
        
        if not csv_file_path:
            error_msg = f"CSV file not found. Searched paths: {possible_paths}"
            print(f"DEBUG ERROR: {error_msg}")
            return JsonResponse({
                "error": error_msg,
                "debug_info": {
                    "base_dir": str(settings.BASE_DIR),
                    "searched_paths": possible_paths
                }
            }, status=404)

        # Load and process data
        try:
            print(f"DEBUG: Attempting to read CSV: {csv_file_path}")
            data = pd.read_csv(csv_file_path)
            print(f"DEBUG: Successfully loaded CSV with {len(data)} rows and columns: {list(data.columns)}")
        except Exception as e:
            print(f"DEBUG ERROR: Error reading CSV file: {e}")
            return JsonResponse({"error": f"Error reading CSV file: {str(e)}"}, status=500)

        if data.empty:
            print("DEBUG ERROR: CSV file is empty")
            return JsonResponse({"error": "CSV file is empty"}, status=404)

        # Clean column names
        original_columns = list(data.columns)
        data.columns = [col.strip().replace(" ", "_").lower() for col in data.columns]
        print(f"DEBUG: Cleaned columns: {original_columns} -> {list(data.columns)}")

        # Handle different possible column names for consistency
        column_mappings = {
            'milk_l': ['milk_l', 'milk_liters', 'milk_yield', 'milk'],
            'feed_kg': ['feed_kg', 'feed', 'feed_consumption'],
            'weight_kg': ['weight_kg', 'weight', 'body_weight'],
            'temperature': ['temperature', 'temp', 'ambient_temp'],
            'humidity': ['humidity', 'relative_humidity', 'rh']
        }

        # Apply column mappings
        for target_col, possible_cols in column_mappings.items():
            for possible_col in possible_cols:
                if possible_col in data.columns and target_col not in data.columns:
                    data[target_col] = data[possible_col]
                    print(f"DEBUG: Mapped column {possible_col} -> {target_col}")
                    break

        # Convert date column safely - FIXED DATE PARSING
        date_columns = ['date', 'timestamp', 'record_date']
        for date_col in date_columns:
            if date_col in data.columns:
                try:
                    # Try multiple date formats to handle different formats in CSV
                    date_formats = ['%d-%m-%Y %H:%M', '%Y-%m-%d %H:%M:%S', '%Y-%m-%d', '%d/%m/%Y %H:%M']
                    
                    for fmt in date_formats:
                        try:
                            data[date_col] = pd.to_datetime(data[date_col], format=fmt, errors='raise')
                            print(f"DEBUG: Successfully converted {date_col} using format: {fmt}")
                            break
                        except:
                            continue
                    else:
                        # If none of the formats work, use coerce to handle errors
                        data[date_col] = pd.to_datetime(data[date_col], errors='coerce')
                        print(f"DEBUG: Used coerce for {date_col} conversion")
                    
                    # Remove rows where date conversion failed
                    before_count = len(data)
                    data = data[data[date_col].notna()]
                    after_count = len(data)
                    print(f"DEBUG: Date conversion complete. Rows: {before_count} -> {after_count}")
                    
                    if date_col != 'date':
                        data['date'] = data[date_col]
                    break
                except Exception as e:
                    print(f"DEBUG WARNING: Could not convert {date_col} to datetime: {e}")

        # Fill missing values with defaults
        numeric_columns = ['milk_l', 'feed_kg', 'weight_kg', 'temperature', 'humidity', 'parity']
        for col in numeric_columns:
            if col in data.columns:
                try:
                    data[col] = pd.to_numeric(data[col], errors='coerce').fillna(0)
                    print(f"DEBUG: Processed numeric column: {col}")
                except Exception as e:
                    print(f"DEBUG WARNING: Could not process column {col}: {e}")

        # Convert to dict for JSON response
        cattle_records = data.to_dict(orient="records")
        print(f"DEBUG: Prepared {len(cattle_records)} records for response")

        # Calculate KPIs safely
        total_cattle = len(cattle_records)
        daily_milk_production = 0
        avg_feed = 0
        
        if 'milk_l' in data.columns:
            try:
                daily_milk_production = float(data['milk_l'].sum())
                print(f"DEBUG: Calculated milk production: {daily_milk_production}")
            except Exception as e:
                print(f"DEBUG WARNING: Could not calculate milk production: {e}")
        
        if 'feed_kg' in data.columns:
            try:
                avg_feed = float(data['feed_kg'].mean())
                print(f"DEBUG: Calculated avg feed: {avg_feed}")
            except Exception as e:
                print(f"DEBUG WARNING: Could not calculate avg feed: {e}")
        
        # Calculate health index based on active animals
        active_animals = 0
        if 'milk_l' in data.columns:
            try:
                active_animals = len(data[data['milk_l'] > 0])
            except:
                active_animals = total_cattle
        
        health_index = (active_animals / total_cattle * 100) if total_cattle > 0 else 0

        kpis = {
            "total_cattle": total_cattle,
            "total_milk": round(daily_milk_production, 1),
            "health_index": round(health_index, 1),
            "avg_feed": round(avg_feed, 2),
        }

        # Milk trend (last 7 days) - Simplified
        milk_trend_labels = []
        milk_trend_data = []
        
        if 'date' in data.columns and 'milk_l' in data.columns:
            try:
                # Get recent data
                recent_dates = data['date'].max() - timedelta(days=7)
                recent_data = data[data['date'] >= recent_dates]
                if len(recent_data) > 0:
                    milk_trend = recent_data.groupby('date')['milk_l'].sum().reset_index()
                    milk_trend_labels = milk_trend['date'].dt.strftime("%Y-%m-%d").tolist()
                    milk_trend_data = milk_trend['milk_l'].tolist()
                    print(f"DEBUG: Generated milk trend with {len(milk_trend_labels)} data points")
                else:
                    print("DEBUG: No recent data found for milk trend")
            except Exception as e:
                print(f"DEBUG WARNING: Could not generate milk trend: {e}")

        # Breed distribution - Simplified
        breed_labels = []
        breed_data = []
        
        if 'breed' in data.columns:
            try:
                breed_counts = data['breed'].value_counts()
                breed_labels = breed_counts.index.tolist()
                breed_data = breed_counts.values.tolist()
                print(f"DEBUG: Generated breed distribution with {len(breed_labels)} breeds")
            except Exception as e:
                print(f"DEBUG WARNING: Could not generate breed distribution: {e}")

        response_data = {
            "kpis": kpis,
            "cattle_records": cattle_records,
            "milk_trend_labels": milk_trend_labels,
            "milk_trend_data": milk_trend_data,
            "breed_labels": breed_labels,
            "breed_data": breed_data,
            "debug_info": {
                "csv_path": csv_file_path,
                "total_rows": len(data),
                "columns": list(data.columns),
                "original_columns": original_columns
            }
        }
        
        print("DEBUG: Successfully prepared dashboard data response")
        return JsonResponse(response_data)

    except Exception as e:
        import traceback
        error_details = {
            "error_type": type(e).__name__,
            "error_message": str(e),
            "traceback": traceback.format_exc(),
            "base_dir": str(settings.BASE_DIR) if 'settings' in globals() else "Unknown"
        }
        
        print(f"DEBUG CRITICAL ERROR: {error_details}")
        
        return JsonResponse({
            "error": f"Internal server error: {str(e)}",
            "debug_info": error_details
        }, status=500)
        