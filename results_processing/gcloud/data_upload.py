from google.cloud import bigquery
import itertools


NEONATES = ['neo007', 'neo021']

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


for neonate in NEONATES:

    print("Working on {}".format(neonate))
    # Construct a BigQuery client object.
    client = bigquery.Client.from_service_account_json(
    "./hypothermia-auth.json"
)

    # Set dataset_id to the ID of the dataset to create.

    dataset_id = 'neo_desat'
    table_id = neonate+"_gradient"

    filename = '/home/buck06191/Dropbox/phd/desat_neonate/ABC/gradient_SA/{}/all_parameters.csv'.format(
        neonate)

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
