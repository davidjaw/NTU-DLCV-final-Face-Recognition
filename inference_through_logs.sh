for name in "$@"
do
  python3 test_validation.py --model_name "teacher.ckpt-$name"
done

