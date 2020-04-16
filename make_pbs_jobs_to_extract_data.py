#!/usr/bin/env python
"""
Script to make NCI .pbs scripts to run extraction for shapefiles

After running this script can submit jobs (has to be on gadi login) using:

for jfile in `ls *pbs`; do qsub $PWD/$jfile; done

"""

import os
import glob

# Output directory
out_dir = "/g/data/r78/LCCS_Aberystwyth/training_data/2010_extracted_03042020"

# Path to extraction script (assume there is a copy in output directory)
extraction_script = os.path.join(out_dir, "extract_data_for_shp.py")

# Get a list of shapefiles
shp_list = glob.glob("/g/data/r78/LCCS_Aberystwyth/training_data/2010/*shp")

# Run for multiple products
for product in ["ls5_nbart_geomedian_annual", "ls5_nbart_tmad_annual"]:
    for shp in shp_list:
        shp_basename = os.path.splitext(os.path.basename(shp))[0]
        out_jobs_script = os.path.join(out_dir, "{}_{}_job.pbs".format(shp_basename, product))
        out_geomedian_txt = os.path.join(out_dir, "{}_{}_stats.txt".format(shp_basename, product))
        out_mads_txt = os.path.join(out_dir, "{}_mads_stats.txt".format(shp_basename))

        out_job_text = """
#PBS -q normal
#PBS -P u46
#PBS -l mem=32GB
#PBS -l ncpus=1
#PBS -l walltime=24:00:00
#PBS -l wd
#PBS -l storage=gdata/v10+gdata/r78+gdata/rs0+gdata/u46+gdata/fk4

module use /g/data/v10/public/modules/modulefiles
module load dea

python {extraction_script} --product {product} -o {out_geomedian_txt} --year 2010 {in_shp}

    """.format(extraction_script=extraction_script, in_shp=shp,
               out_geomedian_txt=out_geomedian_txt, out_dir=out_dir,
               out_mads_txt=out_mads_txt, product=product)

        with open(out_jobs_script, "w") as f:
            f.write(out_job_text)

        print("Wrote to: {}".format(out_jobs_script))
