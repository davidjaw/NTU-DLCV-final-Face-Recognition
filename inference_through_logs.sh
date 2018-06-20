for name in "$@"
do
  python3 test_validation.py --model_name "student.ckpt-$name"
done

