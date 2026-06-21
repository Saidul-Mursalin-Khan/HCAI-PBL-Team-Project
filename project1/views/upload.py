import pandas as pd
from django.shortcuts import render, redirect
from ..models import UploadedDataset
from .common import MAX_UPLOAD_SIZE, read_csv_dataset

def upload(request):
    error = None
    if request.method == 'POST' and request.FILES.get('dataset'):
        uploaded_file = request.FILES['dataset']

        # Validate CSV
        if not uploaded_file.name.lower().endswith('.csv'):
            error = "Only CSV files are supported."
        elif uploaded_file.size > MAX_UPLOAD_SIZE:
            error = "The uploaded CSV is larger than 50 MB."
        else:
            # Save to DB & disk
            obj = UploadedDataset(file=uploaded_file)
            obj.save()

            file_path = obj.file.path  # absolute path on disk

            try:
                df = read_csv_dataset(file_path)
            except ValueError as exc:
                obj.delete()
                return render(request, 'project1/upload.html', {'error': str(exc)})

            # Store in session
            request.session['dataset_path'] = file_path

            # Reset old experiment state
            request.session.pop('target_column', None)
            request.session.pop('problem_type', None)
            request.session.pop('split_path', None)
            request.session.pop('test_size', None)
            request.session['training_completed'] = False
            request.session['columns'] = list(df.columns)

            # Pass table preview to template
            table_html = df.head(20).to_html(
                classes='data-table', index=False, border=0
            )

            return render(request, 'project1/upload.html', {
                'table_html': table_html,
                'columns': list(df.columns),
                'shape': df.shape,
            })

    return render(request, 'project1/upload.html', {'error': error})
