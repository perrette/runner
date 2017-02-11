# dummy model for testing
if [ $1 ]; then
    RUNDIR=$1
fi
if [ $2 ]; then
    a=$2
fi
if [ $3 ]; then
    b=$3
fi
echo "RUNDIR $RUNDIR"
echo "a $a"
echo "b $b"
echo "a $a" > $RUNDIR/output
echo "b $b" >> $RUNDIR/output
