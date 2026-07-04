import os
import pickle
import tempfile

from django.core.files.uploadedfile import SimpleUploadedFile
from django.test import RequestFactory, TestCase


class Project1WorkflowTests(TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.settings_override = self.settings(MEDIA_ROOT=self.tmpdir.name)
        self.settings_override.enable()
        self.factory = RequestFactory()

    def tearDown(self):
        self.settings_override.disable()
        self.tmpdir.cleanup()

    def make_request(self, method, path, data=None, session=None):
        request_method = getattr(self.factory, method)
        request = request_method(path, data=data or {})
        request.session = session if session is not None else {}
        return request

    def upload_csv(self, content, name="dataset.csv"):
        from project1.views.upload import upload

        file_obj = SimpleUploadedFile(name, content.encode("utf-8"), content_type="text/csv")
        request = self.make_request("post", "/project1/upload/", {"dataset": file_obj})
        response = upload(request)
        return request.session, response

    def configure(self, session, target="target", problem_type="classification", test_size="0.25"):
        from project1.views.configure import configure

        request = self.make_request("post", "/project1/configure/", {
            "target_column": target,
            "problem_type": problem_type,
            "test_size": test_size,
        }, session=session)
        response = configure(request)
        return request.session, response

    def assert_response_contains(self, response, text):
        self.assertIn(text, response.content.decode("utf-8"))

    def test_upload_rejects_non_csv(self):
        from project1.views.upload import upload

        file_obj = SimpleUploadedFile("dataset.txt", b"a,b\n1,2\n", content_type="text/plain")
        request = self.make_request("post", "/project1/upload/", {"dataset": file_obj})

        response = upload(request)

        self.assertEqual(response.status_code, 200)
        self.assert_response_contains(response, "Only CSV files are supported.")

    def test_upload_accepts_csv_and_sets_session(self):
        session, response = self.upload_csv("feature,target\n1,A\n2,B\n")

        self.assertEqual(response.status_code, 200)
        self.assert_response_contains(response, "Data Preview")
        self.assertEqual(session["columns"], ["feature", "target"])

    def test_configure_preprocesses_missing_and_categorical_values(self):
        session, _ = self.upload_csv(
            "age,color,target\n"
            "21,red,A\n"
            ",blue,B\n"
            "35,,A\n"
            "42,red,B\n"
            "30,blue,A\n"
            "28,red,B\n"
            "31,green,A\n"
            "44,blue,B\n"
        )

        session, response = self.configure(session)

        self.assertEqual(response.status_code, 302)
        self.assertEqual(response.url, "/project1/visualize/")
        split_path = session["split_path"]
        self.assertTrue(os.path.exists(split_path))
        with open(split_path, "rb") as handle:
            split = pickle.load(handle)
        self.assertIn("X_train", split)
        self.assertGreater(split["X_train"].shape[1], 1)

    def test_visualize_handles_dataset_with_no_numeric_columns(self):
        from project1.views.visualize import visualize

        session, _ = self.upload_csv(
            "text,source,target\n"
            "good,web,yes\n"
            "bad,email,no\n"
            "ok,web,yes\n"
            "fine,email,no\n"
        )
        session, _ = self.configure(session, target="target", problem_type="classification", test_size="0.5")
        request = self.make_request("get", "/project1/visualize/", session=session)

        response = visualize(request)

        self.assertEqual(response.status_code, 200)
        self.assert_response_contains(response, "No numeric columns were found")
        self.assert_response_contains(response, "Target Distribution")

    def test_classification_invalid_model_post_returns_form_error(self):
        from project1.views.classification import classification_train

        session, _ = self.upload_csv(
            "feature,target\n"
            "1,A\n"
            "2,B\n"
            "3,A\n"
            "4,B\n"
        )
        session, _ = self.configure(session, target="target", problem_type="classification", test_size="0.5")
        request = self.make_request("post", "/project1/classification/", {"model": "bad_model"}, session=session)

        response = classification_train(request)

        self.assertEqual(response.status_code, 200)
        self.assert_response_contains(response, "Select a valid classification model.")

    def test_small_regression_dataset_uses_na_for_adjusted_r2(self):
        from project1.views.regression import regression_train

        session, _ = self.upload_csv(
            "feature,target\n"
            "1,10\n"
            "2,20\n"
            "3,30\n"
            "4,40\n"
        )
        session, _ = self.configure(session, target="target", problem_type="regression", test_size="0.5")
        request = self.make_request("post", "/project1/regression/", {"model": "linear_regression"}, session=session)

        response = regression_train(request)

        self.assertEqual(response.status_code, 200)
        self.assert_response_contains(response, "Training results")
        self.assert_response_contains(response, "N/A")
