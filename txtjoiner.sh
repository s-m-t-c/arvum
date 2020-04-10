### This bash file joins training data files together and is rurn after the per tile extraction process

# Set up file name variables
FILE1="tmad_annual_stats"
FILE2="geomedian_annual_stats"
FEATURENUM=10
# Grab the first row from teh first file that matches the variable
ls *$FILE1* | head -1 | xargs head -n +1 > ${FILE1}_joined.txt
# Append onto the above file the contents of files that match the variable whilst skipping the first row
tail -n +2 -q *${FILE1}.txt >> ${FILE1}_joined.txt
ls *$FILE2* | head -1 | xargs head -n +1 > ${FILE2}_joined.txt
tail -n +2 -q *${FILE2}.txt >> ${FILE2}_joined.txt
# Join both files by column
pr -mts' ' *joined.txt > training_data.txt
# Manually alter the first row so columns are aligned
# Remove the duplicated column and any rows that do not have the requisite number of columns (features)
awk '{$8="";print $0}' training_data.txt | sed 's/  / /' | awk 'NF==${FEATURENUM}' > training_datatrim.txt
rm training_data.txt ${FILE1}_joined.txt ${FILE2}_joined.txt
# NOTE this can sometimes make a file with an uneven number of columns
#if there aren't pairs for each of FILE1 and FILE2
