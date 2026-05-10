def stepper_context(request):
    path = request.path

    step_items = [
        (1, 'Upload & View Dataset'),
        (2, 'Configure'),
        (3, 'Data Visualization'),
        (4, 'Training & Result'),
    ]

    training_completed = request.session.get('training_completed', False)

    if 'configure' in path:
        active_step = 2

    elif 'visualize' in path:
        active_step = 3

    elif 'classification' in path or 'regression' in path:

        # After training → mark all completed
        if training_completed:
            active_step = 5
        else:
            active_step = 4

    else:
        active_step = 1

    return {
        'step_items': step_items,
        'active_step': active_step,
    }