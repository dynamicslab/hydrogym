# !/bin/bash
repo_name="yw_dev"
branch_name="yw/hydrogym_v1.0_nekTest"
# Update the repository
git add .
git commit -m "Update $repo_name the repository via branch $branch_name"
git push --force $repo_name $branch_name
