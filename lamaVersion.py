import pkg_resources
import streamlit as st

def get_llma_version():
    try:
        lama_version = pkg_resources.get_distribution("lama").version
        print("LAMA version:", lama_version)
        return lama_version
    except pkg_resources.DistributionNotFound:
        print("LAMA is not installed.")
        return "Not Valid"


def main():
    get_llma_version()

if __name__ == "__main__":
    main()


