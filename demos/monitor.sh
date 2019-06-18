#!/bin/sh
# run the command with all arguments
$@ &
# get pid
PID=$(echo $!)
rm top.dat
while true
do
# get memory usage
RAM=$(sudo pmap $PID | tail -n 1 | awk '/[0-9]K/{print $2}')
# if null, then exit
if test -z $RAM
then
  echo "Exiting..."
  exit
fi
# get current time
DATE=$(date +%r)
echo $DATE'\t'$RAM >> top.dat
# sleep 1 second
sleep 1
done

