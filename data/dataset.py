import os
from collections.abc import Sequence
import numpy as np
import pandas as pd
import json
class Physionet2012DataReader(Sequence):
    """Reader class for physionet 2012 dataset."""

    static_features = [
        'Age', 'Gender', 'ICUType', 'Height'
    ]
    ts_features = [
        'Weight', 'ALP', 'ALT', 'AST', 'Albumin', 'BUN', 'Bilirubin',
        'Cholesterol', 'Creatinine', 'DiasABP', 'FiO2', 'GCS', 'Glucose',
        'HCO3', 'HCT', 'HR', 'K', 'Lactate', 'MAP', 'MechVent', 'Mg',
        'NIDiasABP', 'NIMAP', 'NISysABP', 'Na', 'PaCO2', 'PaO2', 'Platelets',
        'RespRate', 'SaO2', 'SysABP', 'Temp', 'TroponinI', 'TroponinT',
        'Urine', 'WBC', 'pH'
    ]

    categorical_demographics = {
        'Gender': [0, 1],
        'ICUType': [1, 2, 3, 4]
    }
    expanded_static_features = [
        'Age', 'Gender=0', 'Gender=1', 'ICUType=1', 'ICUType=2',
        'ICUType=3', 'ICUType=4', 'Height'
    ]

    # Remove instances without any timeseries
    def __init__(self, data_paths, endpoint_file, vital_features = None, event_features = None):
        """Load instances from the Physionet 2012 challenge.

        Args:
            data_path: Path contiaing the patient records.
            endpoint_file: File containing the endpoint defentions for patient
                           records.

        """

        meta = json.load(open('./data/physionet/feature_meta.json', 'r'))
        self.vital_features = meta['vital']
        self.event_features = meta['event']

        self.data_paths = data_paths
        endpoint_data = pd.read_csv(endpoint_file, header=0, sep=',')
        self.endpoint_data = endpoint_data[
            ~endpoint_data['RecordID'].isin(self.blacklist)]
        if vital_features != None:
            self.vital_features = vital_features
        if event_features != None:
            self.event_features = event_features

    def _convert_string_to_decimal_time(self, values):
        return values.str.split(':').apply(
            lambda a: float(a[0]) + float(a[1])/60
        )

    def __getitem__(self, index):
        """Get instance at position index of endpoint file."""
        example_row = self.endpoint_data.iloc[index, :]

        # Extract targets and record id
        targets = example_row.to_dict()
        record_id = targets['RecordID']
        del targets['RecordID']
        # Read data
        statics, timeseries = self._read_file(str(record_id))
        # time = timeseries['Time']
        values = timeseries[self.ts_features]

        # return record_id, {
        #     'demographics': statics,
        #     'time': time,
        #     'vitals': values,
        #     'targets': targets,
        #     'metadata': {
        #         'patient_id': int(record_id)
        #     }
        # }
        return record_id, statics, values[self.vital_features].dropna(how='all'), values[self.event_features].dropna(how='all'), targets
        # return record_id, values.to_numpy(), targets['In-hospital_death']

    def _read_file(self, record_id):
        # filename = None
        # for path in self.data_paths:
        #     print(path)
        #     suggested_filename = os.path.join(path, record_id + '.txt')
        #     if suggested_filename in os.listdir(path):
        #         filename = suggested_filename
        #         break
        # print(record_id)
        filename = None
        for path in self.data_paths:
            suggested_filename = '{}.txt'.format(record_id)
            if suggested_filename in os.listdir(path):
                filename = os.path.join(path, suggested_filename)
                break
        if filename is None:
            raise ValueError(f'Unable to find data for record: {record_id}.')

        with open(filename) as f:
            data = pd.read_csv(f, sep=',', header=0)

        # Convert time to hours
        # data['Time'] = self._convert_string_to_decimal_time(data['Time'])
        data['Time'] = data['Time'] + ':00'

        # Extract statics
        statics_indicator = data['Parameter'].isin(
            ['RecordID'] + self.static_features)
        statics = data[statics_indicator]
        data = data[~statics_indicator]

        # Handle duplicates in statics
        duplicated_statics = statics[['Time', 'Parameter']].duplicated()
        if duplicated_statics.sum() > 0:
            # logging.warning('Got duplicated statics: %s', statics)
            # Average over duplicate measurements
            statics = statics.groupby(['Time', 'Parameter'], as_index=False)\
                .mean().reset_index()
        statics = statics.pivot(
            index='Time', columns='Parameter', values='Value')
        statics = statics.reindex().reset_index()
        statics = statics.iloc[0]

        # Be sure we are loading the correct record
        assert str(int(statics['RecordID'])) == record_id
        # Drop RecordID
        statics = statics[self.static_features]

        # Do one hot encoding for categorical features
        for demo, values in self.categorical_demographics.items():
            cur_demo = statics[demo]
            # Transform categorical values into zero based index
            to_replace = {val: values.index(val) for val in values}
            # Ensure we dont have unexpected values
            if cur_demo in to_replace.keys():
                indicators = to_replace[cur_demo] #.replace(to_replace).values
                one_hot_encoded = np.eye(len(values))[indicators]
            else:
                # We have a few cases where the categorical variables are not
                # available. Then we should just return zeros for all
                # categories.
                one_hot_encoded = np.zeros(len(to_replace.values()))
            statics.drop(columns=demo, inplace=True)
            columns = [f'{demo}={val}' for val in values]
            statics = pd.concat([statics, pd.Series(one_hot_encoded, index=columns)])

        # Ensure same order
        statics = statics[self.expanded_static_features]

        # Sometimes the same value is observered twice for the same time,
        # potentially using different devices. In this case take the mean of
        # the observed values.
        duplicated_ts = data[['Time', 'Parameter']].duplicated()
        if duplicated_ts.sum() > 0:
            # logging.debug(
            #     'Got duplicated time series variables for RecordID=%s',
            #     record_id
            # )
            data = data.groupby(['Time', 'Parameter'], as_index=False)\
                .mean().reset_index()

        time_series = data.pivot(
            index='Time', columns='Parameter', values='Value')
        time_series = time_series\
            .reindex(columns=self.ts_features).dropna(how='all').reset_index().set_index('Time')
        time_series.index = pd.to_timedelta(time_series.index)
        return statics, time_series

    def __len__(self):
        """Return number of instances in the dataset."""
        return len(self.endpoint_data)

class MIMICnonvital(Sequence):
    def __init__(self, path, ) -> None:
        super().__init__()