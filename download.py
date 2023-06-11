import click
import json
import urllib
import urllib.request


@click.command()
@click.option('--json_file_name', default='MSCOCO_train_val_Korean.json', help='Json file name')
@click.option('--data_dir', default='data/', help='Data directory')
@click.option('--data_size', default=120000, help='Data size')
def run(json_file_name, data_dir, data_size):
    with open(json_file_name, 'r', encoding='UTF-8') as f:
        json_data = json.load(f)

    # Download data
    for i in range(data_size):
        if i % 100 == 0:
            print(i, '/', data_size)

        try:
            id = json_data[i]['file_path']
            name = id[-16:]
            url = 'http://images.cocodataset.org/' + id
            urllib.request.urlretrieve(url, data_dir + name)
        except Exception as e:
            print(e)
    # time.sleep(0.1)


if __name__ == '__main__':
    run()