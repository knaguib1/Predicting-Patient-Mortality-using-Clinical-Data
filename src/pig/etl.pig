-- ***************************************************************************
-- TASK
-- Aggregate events into features of patient and generate training, testing data for mortality prediction.
-- Steps have been provided to guide you.
-- You can include as many intermediate steps as required to complete the calculations.
-- You can NOT touch the provided load path or variable names because it will be used in grading.
-- ***************************************************************************

-- ***************************************************************************
-- TESTS
-- To test, you can change the LOAD path for events and mortality to ../../test/events.csv and ../../test/mortality.csv, however, you MUST switch it back to original path in submission.
-- 6 tests have been provided to test all the subparts in this exercise.
-- Manually compare the output of each test against the csv's in test/expected folder.
-- ***************************************************************************

-- register a python UDF for converting data into SVMLight format
REGISTER utils.py USING jython AS utils;

-- load events file
-- UNCOMMENT BOTTOM LINE WHEN READY FOR SUBMITTION
events = LOAD '../../data/events.csv' USING PigStorage(',') AS (patientid:int, eventid:chararray, eventdesc:chararray, timestamp:chararray, value:float);
--events = LOAD '../sample_test/sample_events.csv' USING PigStorage(',') AS (patientid:int, eventid:chararray, eventdesc:chararray, timestamp:chararray, value:float);

-- select required columns from events
events = FOREACH events GENERATE patientid, eventid, ToDate(timestamp, 'yyyy-MM-dd') AS etimestamp, value;

-- load mortality file
-- UNCOMMENT BOTTOM LINE WHEN READY FOR SUBMITTION
mortality = LOAD '../../data/mortality.csv' USING PigStorage(',') as (patientid:int, timestamp:chararray, label:int);
--mortality = LOAD '../sample_test/sample_mortality.csv' USING PigStorage(',') as (patientid:int, timestamp:chararray, label:int);
mortality = FOREACH mortality GENERATE patientid, ToDate(timestamp, 'yyyy-MM-dd') AS mtimestamp, label;

--To display the relation, use the dump command e.g. DUMP mortality;

-- ***************************************************************************
-- Compute the index dates for dead and alive patients
-- ***************************************************************************
-- perform join of events and mortality by patientid;
eventswithmort = JOIN events BY patientid, mortality BY patientid;

-- detect the events of dead patients and create it of the form (patientid, eventid, value, label, time_difference) where time_difference is the days between index date and each event timestamp
-- INDEX_DATE: For deceased patients: Index date is 30 days prior to the death date (timestampfield) in mortality.csv
-- Use "SubtractDuration" to subtract 30 days from death date
-- USE "DaysBetween" for # days between death date and event ts. 
deadevents = FOREACH eventswithmort GENERATE events::patientid, events::eventid, events::value, mortality::label, DaysBetween(SubtractDuration(mortality::mtimestamp,'P30D'), events::etimestamp) as time_difference;

-- detect the events of alive patients and create it of the form (patientid, eventid, value, label, time_difference) where time_difference is the days between index date and each event timestamp
-- For alive patients: Index date is the last event date in events.csv for each alive patient.

all_patientid = FOREACH events GENERATE patientid;
dead_patientid = FOREACH mortality GENERATE patientid;
mortality_mapping = JOIN all_patientid BY patientid LEFT OUTER, dead_patientid BY patientid;
alive_patients = FILTER mortality_mapping by dead_patientid::patientid is null;
alive_patients = FOREACH alive_patients GENERATE $0 AS patientid;
alive_patients = DISTINCT alive_patients;
alive_temp = JOIN events BY patientid, alive_patients BY patientid;
alive_temp = FOREACH alive_temp GENERATE $0 AS patientid, $1 AS eventid, $2 AS etimestamp, $3 AS value; 

-- Need to find max events ts for each patient, 
alive_ts = FOREACH alive_temp GENERATE $0 AS patientid, $2 AS timestamp;
alive_group = GROUP alive_ts BY patientid;
alive_maxts = FOREACH alive_group {
                sortByMax = ORDER alive_ts BY timestamp DESC;
                topMax = LIMIT sortByMax 1;
                GENERATE FLATTEN(topMax.$0) AS patientid, FLATTEN(topMax.$1) AS mtimestamp;
                };
alive_temp = JOIN events BY patientid, alive_maxts BY patientid;
alive_temp = FOREACH alive_temp GENERATE $0 AS patientid, $1 AS eventid, $3 AS value, 0 AS label, $2 AS etimestamp, $5 AS mtimestamp;
aliveevents = FOREACH alive_temp GENERATE patientid, eventid, value, label, DaysBetween(mtimestamp, etimestamp) as time_difference;

--TEST-1
deadevents = ORDER deadevents BY patientid, eventid;
aliveevents = ORDER aliveevents BY patientid, eventid;
--STORE aliveevents INTO 'aliveevents' USING PigStorage(',');
--STORE deadevents INTO 'deadevents' USING PigStorage(',');

-- ***************************************************************************
-- Filter events within the observation window and remove events with missing values
-- Tip: please ensure the number of rows of your output feature is 3618, otherwise the following sections will be highly impacted and make you lose all the relevant credits.
-- ***************************************************************************
-- Consider an observation window (2000 days) and prediction window (30 days). Remove the events that occur outside the observation window.
alive_filtered = FILTER aliveevents BY (time_difference <= 2000) AND (0 <= time_difference) AND (value is not null);
dead_filtered = FILTER deadevents BY (time_difference <= 2000) AND (0 <= time_difference) AND (value is not null);

-- contains only events for all patients within the observation window of 2000 days and is of the form (patientid, eventid, value, label, time_difference)
filtered = UNION alive_filtered, dead_filtered;
--TEST-2
filteredgrpd = GROUP filtered BY 1;
filtered = FOREACH filteredgrpd GENERATE FLATTEN(filtered);
filtered = ORDER filtered BY patientid, eventid,time_difference;
--STORE filtered INTO 'filtered' USING PigStorage(',');

-- ***************************************************************************
-- Aggregate events to create features
-- ***************************************************************************
-- for group of (patientid, eventid), count the number of  events occurred for the patient 
-- and create relation of the form (patientid, eventid, featurevalue)
filtered_grp = GROUP filtered BY (patientid, eventid);
featureswithid = FOREACH filtered_grp GENERATE FLATTEN(group) AS (patientid, eventid), COUNT(filtered.eventid) AS featurevalue;

--TEST-3
featureswithid = ORDER featureswithid BY patientid, eventid;
--STORE featureswithid INTO 'features_aggregate' USING PigStorage(',');

-- ***************************************************************************
-- Generate feature mapping
-- ***************************************************************************
-- compute the set of distinct eventids obtained from previous step, 
-- sort them by eventid and then rank these features by eventid to create (idx, eventid). Rank should start from 0.
feature_names = FOREACH featureswithid GENERATE eventid;
feature_names = DISTINCT feature_names;
feature_names = ORDER feature_names BY eventid ASC;
feature_idx = RANK feature_names;

all_features = FOREACH feature_idx GENERATE ($0-1) AS idx, $1 AS eventid;

-- store the features as an output file
--STORE all_features INTO 'features' using PigStorage(' ');

-- perform join of featureswithid and all_features by eventid and replace eventid with idx. 
-- It is of the form (patientid, idx, featurevalue)
features = JOIN featureswithid by eventid, all_features by eventid;
features = FOREACH features GENERATE $0 AS patientid, $3 AS idx, $2 AS featurevalue;

--TEST-4
features = ORDER features BY patientid, idx;
--STORE features INTO 'features_map' USING PigStorage(',');

-- ***************************************************************************
-- Normalize the values using min-max normalization
-- Use DOUBLE precision
-- ***************************************************************************
-- group events by idx and compute the maximum feature value in each group. It is of the form (idx, maxvalue)
idx_grp = GROUP features by idx;
maxvalues = FOREACH idx_grp GENERATE FLATTEN(features.idx) AS idx, MAX(features.featurevalue)*1.0 AS maxvalue;
maxvalues = DISTINCT maxvalues;

-- join features and maxvalues by idx
normalized = JOIN features by idx, maxvalues by idx;

-- compute the final set of normalized features of the form (patientid, idx, normalizedfeaturevalue)
features = FOREACH normalized GENERATE $0 AS patientid, $1 AS idx, ($2 / $4)*1.0 AS normalizedfeaturevalue;

--TEST-5
features = ORDER features BY patientid, idx;
--STORE features INTO 'features_normalized' USING PigStorage(',');

-- ***************************************************************************
-- Generate features in svmlight format
-- features is of the form (patientid, idx, normalizedfeaturevalue) and is the output of the previous step
-- e.g.  1,1,1.0
--  	 1,3,0.8
--	     2,1,0.5
--       3,3,1.0
-- ***************************************************************************

grpd = GROUP features BY patientid;
grpd_order = ORDER grpd BY $0;
features = FOREACH grpd_order
{
    sorted = ORDER features BY idx;
    generate group as patientid, utils.bag_to_svmlight(sorted) as sparsefeature;
}

-- ***************************************************************************
-- Split into train and test set
-- labels is of the form (patientid, label) and contains all patientids followed by label of 1 for dead and 0 for alive
-- e.g. 1,1
--	    2,0
--      3,1
-- ***************************************************************************

-- create it of the form (patientid, label) for dead and alive patients
labels = FOREACH filtered GENERATE $0 AS patientid, $3 AS label;
labels = DISTINCT labels;

--Generate sparsefeature vector relation
samples = JOIN features BY patientid, labels BY patientid;
samples = DISTINCT samples PARALLEL 1;
samples = ORDER samples BY $0;
samples = FOREACH samples GENERATE $3 AS label, $1 AS sparsefeature;

--TEST-6
--STORE samples INTO 'samples' USING PigStorage(' ');

-- randomly split data for training and testing
DEFINE rand_gen RANDOM('6505');
samples = FOREACH samples GENERATE rand_gen() as assignmentkey, *;
SPLIT samples INTO testing IF assignmentkey <= 0.20, training OTHERWISE;
training = FOREACH training GENERATE $1..;
testing = FOREACH testing GENERATE $1..;

-- save training and tesing data
STORE testing INTO 'testing' USING PigStorage(' ');
STORE training INTO 'training' USING PigStorage(' ');
