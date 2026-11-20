import sqlite3
import tempfile
import unittest
from pathlib import Path

from database_maintenance import backup_database, integrity_check
from database_migrations import apply_migrations, schema_version


class DatabaseMaintenanceTests(unittest.TestCase):
    def test_backup_is_integrity_checked_and_independent(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.db"
            destination = Path(directory) / "backups" / "copy.db"
            connection = sqlite3.connect(source)
            try:
                connection.execute("CREATE TABLE values_table (value TEXT)")
                connection.execute("INSERT INTO values_table VALUES ('before')")
                connection.commit()
            finally:
                connection.close()

            backup_database(source, destination)

            self.assertTrue(integrity_check(source))
            self.assertTrue(integrity_check(destination))
            connection = sqlite3.connect(destination)
            try:
                self.assertEqual(
                    connection.execute(
                        "SELECT value FROM values_table"
                    ).fetchone()[0],
                    "before",
                )
            finally:
                connection.close()

    def test_migrations_record_and_apply_versions_once(self):
        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / "migration.db"
            connection = sqlite3.connect(database)
            try:
                calls = []

                def migration(db):
                    calls.append(True)
                    db.execute("CREATE TABLE marker (value TEXT)")

                migrations = {1: migration}
                self.assertEqual(apply_migrations(connection, migrations), 1)
                self.assertEqual(schema_version(connection), 1)
                self.assertEqual(apply_migrations(connection, migrations), 1)
                self.assertEqual(len(calls), 1)
            finally:
                connection.close()


if __name__ == "__main__":
    unittest.main()
