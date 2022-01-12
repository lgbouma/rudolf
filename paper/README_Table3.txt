================================================================================
Title: A 38 Million Year Old Neptune-Sized Planet in the Kepler Field  
Authors: Bouma L.G., Curtis J.L., Masuda K., Hillenbrand L.A., Stefansson G., 
         Isaacson H., Narita N., Fukui A., Ikoma M., Tamura M., Kraus A.L., 
         Furlan E., Gnilka C.L., Lester K.V., Howell S.B. 
================================================================================
Description of contents: A comma delimited table,
 cdips_targets_v0.6_nomagcut_gaiasources_table.csv, containing the full 
 results of Table 3. Two python scripts, example_read_script_*.py,
 are also included to help read the table.

System requirements: None for the table but the example scripts 
 require Python 3.

Additional comments: The columns are

  1: Gaia DR2 source identifier
  2: Gaia DR2 Right Ascension (J2015); deg
  3: Gaia DR2 Declination (J2015); deg
  4: Gaia DR2 parallax; mas
  5: Gaia DR2 parallax uncertainty; mas
  6: Gaia DR2 proper motion along RA * cos(Dec); mas/yr
  7: Gaia DR2 proper motion along Dec; mas/yr
  8: Gaia DR2 G band magnitude; mag
  9: Gaia DR2 Blue passband magnitude; mag
 10: Gaia DR2 Red passband magnitude; mag
 11: Cluster or group identifier
 12: log_10_ age; [yr]
 13: log_10_ mean age of previous colum ignoring NaN values; [yr]
 14: Provenance of group membership;
 15: ADS bibcode(s) for previous column.

 Note that columns 11, 12, 14, and 15 are all variable-length strings.  They
 are comma-separated in cases where there are multiple sources of group
 membership The pythons scripts use the pandas library to read these
 variable-length columns.

================================================================================
