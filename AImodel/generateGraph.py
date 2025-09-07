import os
import ast
import pandas as pd
import seaborn as sns

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


def parse_tag_list(value):
    if pd.isna(value):
        return []
    if isinstance(value, list):
        return value
    try:
        parsed = ast.literal_eval(str(value))
        if isinstance(parsed, list):
            # Normalize by stripping extra whitespace and lowering case
            return [str(tag).strip() for tag in parsed if str(tag).strip()]
    except Exception:
        pass
    # Fallback: split by comma if it's a simple CSV-like string
    return [s.strip() for s in str(value).split(',') if s.strip()]


def ensure_datetime_month(frame: pd.DataFrame) -> pd.DataFrame:
    # Prefer an existing month label if present; otherwise derive from publishedAt
    if 'publishedAt' in frame.columns:
        frame['publishedAt'] = pd.to_datetime(frame['publishedAt'], errors='coerce')
    if 'month_of_year' in frame.columns and frame['month_of_year'].dtype == object:
        # Sometimes contains labels like "Oct 2023"; use as is
        frame['month_label'] = frame['month_of_year']
    elif 'publishedAt' in frame.columns:
        frame['month_label'] = frame['publishedAt'].dt.to_period('M').astype(str)
    else:
        # Fallback to numeric month and year if available
        if 'month' in frame.columns:
            frame['month_label'] = frame['month'].astype(str)
        else:
            frame['month_label'] = 'Unknown'

    # Build a proper datetime key for ordering (first day of month)
    if 'publishedAt' in frame.columns and frame['publishedAt'].notna().any():
        frame['month_dt'] = frame['publishedAt'].dt.to_period('M').dt.to_timestamp()
    else:
        # Try parsing the textual month_label like "Oct 2023"
        month_dt_try = pd.to_datetime(frame['month_label'], format='%b %Y', errors='coerce')
        if month_dt_try.isna().all():
            month_dt_try = pd.to_datetime(frame['month_label'], errors='coerce')
        frame['month_dt'] = month_dt_try
    return frame


def build_long_form_by_tag_and_month(frame: pd.DataFrame) -> pd.DataFrame:
    tag_col = 'allTags_cleaned_filtered' if 'allTags_cleaned_filtered' in frame.columns else 'allTags_cleaned'
    frame['tag_list'] = frame[tag_col].apply(parse_tag_list)
    # Explode tags
    exploded = frame.explode('tag_list')
    exploded = exploded[~exploded['tag_list'].isna() & (exploded['tag_list'] != '')]
    return exploded


def compute_top_tags(exploded: pd.DataFrame, top_n: int = 10) -> list:
    # Rank tags by total views across all time
    by_tag = exploded.groupby('tag_list', as_index=False)['viewCount'].sum()
    by_tag = by_tag.sort_values('viewCount', ascending=False)
    return by_tag['tag_list'].head(top_n).tolist()


def plot_total_views_likes(exploded: pd.DataFrame, output_path: str, top_tags: list | None = None):
    subset = exploded.copy()
    if top_tags:
        subset = subset[subset['tag_list'].isin(top_tags)]

    agg = (
        subset.groupby(['month_dt', 'month_label', 'tag_list'], as_index=False)
        .agg({'viewCount': 'sum', 'likeCount': 'sum'})
    )

    agg = agg.sort_values('month_dt')
    # Melt to long format for seaborn
    long_df = agg.melt(id_vars=['month_dt', 'month_label', 'tag_list'], value_vars=['viewCount', 'likeCount'],
                       var_name='metric', value_name='count')

    plt.figure(figsize=(12, 7))
    sns.lineplot(data=long_df, x='month_dt', y='count', hue='tag_list', style='metric', markers=True)
    ax = plt.gca()
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    plt.xticks(rotation=45, ha='right')
    plt.title('Total View and Like Count Trends by Tag')
    plt.xlabel('Month')
    plt.ylabel('Total Count')
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    return os.path.basename(output_path)


def plot_avg_engagement_ratio(exploded: pd.DataFrame, output_path: str, top_tags: list | None = None):
    subset = exploded.copy()
    if top_tags:
        subset = subset[subset['tag_list'].isin(top_tags)]

    # Ensure likes_per_view and comments_per_view exist; if not, derive from raw counts
    if 'likes_per_view' not in subset.columns:
        subset['likes_per_view'] = (subset['likeCount'].fillna(0) /
                                    subset['viewCount'].replace({0: pd.NA}))
    if 'comments_per_view' not in subset.columns:
        subset['comments_per_view'] = (subset['commentCount'].fillna(0) /
                                       subset['viewCount'].replace({0: pd.NA}))
    subset['likes_per_view'] = subset['likes_per_view'].fillna(0)
    subset['comments_per_view'] = subset['comments_per_view'].fillna(0)

    agg = (
        subset.groupby(['month_dt', 'month_label', 'tag_list'], as_index=False)
        .agg({'likes_per_view': 'mean', 'comments_per_view': 'mean'})
    )
    agg = agg.sort_values('month_dt')

    long_df = agg.melt(id_vars=['month_dt', 'month_label', 'tag_list'],
                       value_vars=['likes_per_view', 'comments_per_view'],
                       var_name='metric', value_name='ratio')
    long_df = long_df[long_df['ratio'] <= 1]
    plt.figure(figsize=(12, 7))
    # Handle multiple tags: use both tag_list and metric for proper grouping
    if len(top_tags) == 1:
        # Single tag: show likes vs comments as separate lines
        sns.lineplot(data=long_df, x='month_dt', y='ratio', hue='metric', markers=True)
    else:
        # Multiple tags: combine tag and metric for unique hue values
        long_df['tag_metric'] = long_df['tag_list'] + ' - ' + long_df['metric']
        sns.lineplot(data=long_df, x='month_dt', y='ratio', hue='tag_metric', markers=True)
    
    ax = plt.gca()
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    plt.xticks(rotation=45, ha='right')
    if len(top_tags) == 1:
        plt.title(f'Average Engagement Trends for #{top_tags[0]} (Likes/View vs Comments/View)')
    else:
        plt.title('Average Engagement Trends by Tag (Likes/View vs Comments/View)')
    plt.xlabel('Month')
    plt.ylabel('Average Ratio')
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    return os.path.basename(output_path)

def main(tags):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(base_dir, 'cleanedData.csv')
    # Build a relative output path (relative to current working directory)
    output_dir_abs = os.path.normpath(os.path.join(base_dir, '..', 'docs', 'public'))
    output_dir = os.path.relpath(output_dir_abs, start=os.getcwd())
    os.makedirs(output_dir, exist_ok=True)

    df = pd.read_csv(csv_path)
    # Coerce numeric columns just in case
    for col in ['viewCount', 'likeCount', 'commentCount', 'likes_per_view', 'comments_per_view']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    df = ensure_datetime_month(df)
    exploded = build_long_form_by_tag_and_month(df)

    # Plot 1: total views and likes trends
    output_path1 = plot_total_views_likes(
        exploded,
        output_path=os.path.join(output_dir, 'total_views_likes_by_tag.png'),
        top_tags=tags,
    )

    # Plot 2: average engagement ratio trends
    output_path2 = plot_avg_engagement_ratio(
        exploded,
        output_path=os.path.join(output_dir, 'avg_engagement_ratio_by_tag.png'),
        top_tags=tags,
    )

    return output_path1, output_path2


if __name__ == '__main__':
    selected_tag = ['hair']
    main(selected_tag)

