# !/bin/bash
repo_name="yw_dev"
branch_name="yw/hydrogym_v1.0_nekTest"

commitMessage="Update"  # Default commit message
# Parse arguments for commit message
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --m) commitMessage="$2"; shift ;; # Set commit message
        *) echo "Unknown parameter: $1"; exit 1 ;;
    esac
    shift
done

# Update the repository
git add .
git commit -m"${commitMessage}"
git push --force $repo_name $branch_name
