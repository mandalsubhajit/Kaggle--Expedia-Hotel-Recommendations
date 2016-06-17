@echo off

set ntimes=1

if not "%2"=="" (
 set ntimes=%2
)

echo the loop will run %ntimes% times

for /l %%x in (1, 1, %ntimes%) do (
   echo Epoch %%x Start
   python %1
)