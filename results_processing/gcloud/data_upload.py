from google.cloud import bigquery
import itertools


DATASETS = ['LWP475', 'LWP479', 'LWP481', 'LWP484']
MODELS = ['bph0']
LOCATION = 'EU'


def dataset_exists(client, dataset_reference):
    """Return if a dataset exists.

    Args:
        client (google.cloud.bigquery.client.Client):
            A client to connect to the BigQuery API.
        dataset_reference (google.cloud.bigquery.dataset.DatasetReference):
            A reference to the dataset to look for.

    Returns:
        bool: ``True`` if the dataset exists, ``False`` otherwise.
    """
    from google.cloud.exceptions import NotFound

    try:
        client.get_dataset(dataset_reference)
        return True
    except NotFound:
        return False


combinations = list(itertools.product(DATASETS, MODELS))

for combo in combinations:

    print("Working on {0} - {1}".format(*combo))
    # Construct a BigQuery client object.
    client = bigquery.Client()

    # Set dataset_id to the ID of the dataset to create.
    dataset_id = combo[0]
    table_id = combo[1]

    filename = '/home/buck06191/Dropbox/phd/hypothermia/ABC/nrmse_SA/{}/{}/all_parameters.csv'.format(
        table_id, dataset_id)

    if not dataset_exists(client, dataset_id):
        # Construct a full Dataset object to send to the API.
        dataset = bigquery.Dataset("{}.{}".format(client.project, dataset_id))

        dataset.location = LOCATION

        # Send the dataset to the API for creation.
        # Raises google.api_core.exceptions.Conflict if the Dataset already
        # exists within the project.

        dataset = client.create_dataset(dataset)  # API request
        print("Created dataset {}.{}".format(
            client.project, dataset.dataset_id))

    dataset_ref = client.dataset(dataset_id)
    table_ref = dataset_ref.table(table_id.replace('.', '_'))

    job_config = bigquery.LoadJobConfig()
    job_config.source_format = bigquery.SourceFormat.CSV
    job_config.skip_leading_rows = 1
    job_config.autodetect = True
    print("\tUploading local data.")
    with open(filename, "rb") as source_file:
        job = client.load_table_from_file(
            source_file,
            table_ref,
            location=LOCATION,  # Must match the destination dataset location.
            job_config=job_config,
        )  # API request

    job.result()  # Waits for table load to complete.

    print("\tLoaded {:,} rows into {}:{}.".format(
        job.output_rows, dataset_id, table_id))
