# A csomag közvetlen futtatásakor elindítja az SKF-kiértékelőt.

from .skf_kiertekeles import main

if __name__ == "__main__":
    raise SystemExit(main())
