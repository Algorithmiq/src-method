# Setting up SonarCloud

Adding SonarCloud analysis for `src_method` is done in two steps.

First, ask a software engineer to enable the SonarCloud app for you repository.
This is done through Algorithmiq’s Github organization settings:

- Go to [Algorithmiq’s Github organization settings](https://github.com/organizations/Algorithmiq/settings/profile)
- Click on "GitHub Apps" on the left-hand vertical bar
- Click "Configure" for Sonarcloud
- In the "Repository Access" menu, add your repo `src_method` to the list, to make it visible to the app.
- Click "Save"

Then, on the Sonarcloud dashboard:

- Go to the [Algorithmiq organization](https://sonarcloud.io/organizations/algorithmiq/projects)
- Click on the **+** on the top right corner and choose **Analyze new project**
- Add the `src_method` repo from the list

!!! warning
    We use Sonarcloud through the CI: the analysis method (accessible from **Administration > Analysis method**) must **not** be set to Automatic Analysis.

Finally, you need to set the `SONAR_TOKEN` for your repository:

- Copy the token from [here](https://sonarcloud.io/project/configuration/GitHubActions?id=Algorithmiq_src_method)
- Go to [the **Secrets and variables** menu for actions](https://github.com/Algorithmiq/src-method/settings/secrets/actions)
- Click on **New repository secret**
- Type `SONAR_TOKEN` as name and paste the value you copied from the SonarCloud settings.
- Repeat the same steps in [the **Secrets and variables** menu for Dependabot](https://github.com/Algorithmiq/src-method/settings/secrets/dependabot)

Now the repo is added and reports can be seen from the Sonarcloud dashboard and automatically generated with the Github action.
