from os import environ

DB_CONNECTION_STRING = f"postgresql+psycopg2://" \
                    f"{environ.get('POSTGRES_USER', 'postgres')}:" \
                    f"{environ.get('POSTGRES_PWD', 'postgres')}" \
                    f"@{environ.get('POSTGRES_HOST', 'localhost')}" \
                    f":{environ.get('POSTGRES_PORT', '5432')}/" \
                    f"{environ.get('POSTGRES_DB', 'pg')}"