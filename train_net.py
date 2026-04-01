from test_net import main
from detectron2.engine import default_argument_parser, launch


if __name__ == "__main__":
    args_parser = default_argument_parser()
    args_parser.add_argument(
        "--inference-only",
        action="store_true",
        help="Only run inference on the model.",
    )
    args = args_parser.parse_args()

    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
