if __name__ == "__main__":
    import sys
    import os
    args = sys.argv[1:]
    if len(args) == 0:
        config_dir = os.getcwd() + '/dias.conf'
    elif len(args) != 1:
        print("TODO: usage")
        exit(1)
    else:
        config_dir = args[0]

    print('Loading global config from ' + config_dir)

    from dias import service
    exit_code = service.service(config_dir) or 0
    exit(exit_code)
else:
    raise ImportError("Run this file directly, don't import it!")